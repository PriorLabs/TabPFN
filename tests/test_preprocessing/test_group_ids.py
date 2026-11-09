#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for `group_col`: selecting the columns, coding the keys, using the feature."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.errors import TabPFNValidationError
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.group_ids import GroupIdEncoder, mask_unseen_groups

ESTIMATORS = [TabPFNClassifier, TabPFNRegressor]


def _frame(groups: list[Any], *, column: str = "g") -> pd.DataFrame:
    """A numeric column next to the group column."""
    return pd.DataFrame({"num": np.arange(len(groups), dtype=float), column: groups})


def _estimator_data(
    estimator_cls: type, groups: list[Any]
) -> tuple[pd.DataFrame, np.ndarray]:
    """A frame with a group column, plus a `y` matching the estimator."""
    n = len(groups)
    rng = np.random.default_rng(seed=42)
    X = pd.DataFrame(
        {"num": rng.normal(size=n), "other": rng.normal(size=n), "g": groups}
    )
    y = (
        rng.integers(0, 2, size=n)
        if estimator_cls is TabPFNClassifier
        else rng.normal(size=n)
    )
    return X, y


def _fitted(estimator_cls: type, groups: list[Any], **kwargs: Any):  # noqa: ANN202
    X, y = _estimator_data(estimator_cls, groups)
    estimator = estimator_cls(n_estimators=1, **kwargs)
    estimator.fit(X, y)
    return estimator, X, y


def _group_feature(estimator: Any) -> tuple[int, Any]:
    """The encoded feature's position in the fitted schema, and the feature."""
    name = estimator.group_encoder_.feature_name_
    features = estimator.inferred_feature_schema_.features
    index = [f.name for f in features].index(f"input_{name}")
    return index, features[index]


class TestSelection:
    """What `group_col` selects, and what it refuses."""

    @pytest.mark.parametrize("group_col", [None, []], ids=["none", "empty"])
    def test__nothing_selected__passes_every_input_through(
        self, group_col: Any
    ) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        encoder = GroupIdEncoder(group_col)

        assert encoder.fit_transform(X) is X
        assert encoder.transform(X) is X
        assert encoder.output_indices([1]) == [1]
        assert encoder.output_indices(None) is None

    def test__single_label__is_read_as_one_column(self) -> None:
        encoder = GroupIdEncoder("g")
        encoder.fit(_frame(["a", "b"]))

        assert encoder.group_columns_ == ["g"]

    def test__non_dataframe_input__is_refused(self) -> None:
        with pytest.raises(TabPFNValidationError, match="has to be a DataFrame"):
            GroupIdEncoder("g").fit_transform(np.zeros((2, 2)))

    def test__missing_column__is_refused_naming_it(self) -> None:
        with pytest.raises(
            TabPFNValidationError, match=r"not in the input: \['gone'\]"
        ):
            GroupIdEncoder(["g", "gone"]).fit_transform(_frame(["a", "b"]))

    def test__column_label_the_input_repeats__is_refused(self) -> None:
        X = pd.DataFrame([[1.0, "a", "b"]], columns=["num", "g", "g"])

        with pytest.raises(TabPFNValidationError, match="holds more than"):
            GroupIdEncoder("g").fit_transform(X)

    def test__label_named_twice__is_refused(self) -> None:
        with pytest.raises(TabPFNValidationError, match="more than once"):
            GroupIdEncoder(["g", "g"]).fit_transform(_frame(["a", "b"]))

    @pytest.mark.parametrize(
        "group_col",
        [1, 1.0, {"g"}, ["g", 1], np.array(["g"])],
        ids=["int", "float", "set", "mixed_list", "array"],
    )
    def test__anything_but_labels__is_refused(self, group_col: Any) -> None:
        with pytest.raises(TabPFNValidationError, match="must be a column label"):
            GroupIdEncoder(group_col).fit_transform(_frame(["a", "b"]))

    def test__transform_before_fit__is_refused(self) -> None:
        with pytest.raises(NotFittedError):
            GroupIdEncoder("g").transform(_frame(["a", "b"]))

    def test__predict_input_missing_the_column__is_refused(self) -> None:
        encoder = GroupIdEncoder("g")
        encoder.fit_transform(_frame(["a", "b"]))

        with pytest.raises(TabPFNValidationError, match="not in the input"):
            encoder.transform(pd.DataFrame({"num": [1.0]}))


class TestEncoding:
    """The codes themselves: one per distinct key, fitted on the training rows."""

    def test__repeated_and_singleton_groups__each_keep_a_code(self) -> None:
        out = GroupIdEncoder("g").fit_transform(_frame(["a", "b", "a", "c"]))

        assert list(out.columns) == ["num", "group_id"]
        np.testing.assert_array_equal(out["group_id"], [0.0, 1.0, 0.0, 2.0])

    @pytest.mark.parametrize(
        "groups",
        [["a", "b", "a"], [10, 20, 10], [1.5, 2.5, 1.5]],
        ids=["string", "integer", "float"],
    )
    def test__keys_of_any_type__are_coded_the_same_way(self, groups: list) -> None:
        out = GroupIdEncoder("g").fit_transform(_frame(groups))

        np.testing.assert_array_equal(out["group_id"], [0.0, 1.0, 0.0])

    def test__several_columns__are_one_composite_key(self) -> None:
        X = pd.DataFrame(
            {
                "site": ["s1", "s1", "s2", "s2"],
                "patient": ["p1", "p2", "p1", "p2"],
                "num": [1.0, 2.0, 3.0, 4.0],
            }
        )

        out = GroupIdEncoder(["site", "patient"]).fit_transform(X)

        assert list(out.columns) == ["num", "group_id"]
        # Four distinct pairs, though each column holds only two values: one
        # feature per column would have coded this as two features of two codes.
        np.testing.assert_array_equal(out["group_id"], [0.0, 1.0, 2.0, 3.0])

    def test__unseen_key__is_missing_at_predict(self) -> None:
        encoder = GroupIdEncoder("g")
        encoder.fit_transform(_frame(["a", "a", "b"]))

        out = encoder.transform(_frame(["b", "zzz", "a"]))

        np.testing.assert_array_equal(out["group_id"], [1.0, np.nan, 0.0])

    @pytest.mark.parametrize(
        ("fit_keys", "predict_keys"),
        [([1.0, np.nan], [np.nan]), (["a", None], [None])],
        ids=["nan", "none"],
    )
    def test__missing_key__is_one_group_coded_alike_at_fit_and_predict(
        self, fit_keys: list, predict_keys: list
    ) -> None:
        encoder = GroupIdEncoder("g")
        out = encoder.fit_transform(_frame(fit_keys))

        predicted = encoder.transform(_frame(predict_keys))

        np.testing.assert_array_equal(out["group_id"], [0.0, 1.0])
        np.testing.assert_array_equal(predicted["group_id"], [1.0])

    def test__predict__leaves_the_fitted_mapping_alone(self) -> None:
        encoder = GroupIdEncoder("g")
        encoder.fit_transform(_frame(["a", "b"]))
        before = encoder.keys_.tolist()

        encoder.transform(_frame(["c", "d"]))

        assert encoder.keys_.tolist() == before

    def test__refit__rebuilds_the_mapping(self) -> None:
        encoder = GroupIdEncoder("g")
        encoder.fit_transform(_frame(["a", "b"]))

        out = encoder.fit_transform(_frame(["x", "x", "y"]))

        assert encoder.keys_.tolist() == [("x",), ("y",)]
        np.testing.assert_array_equal(out["group_id"], [0.0, 0.0, 1.0])

    def test__row_order_and_the_callers_frame__are_untouched(self) -> None:
        X = _frame(["a", "b", "a"])
        X.index = [7, 3, 5]

        out = GroupIdEncoder("g").fit_transform(X)

        np.testing.assert_array_equal(out["num"], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(out["group_id"], [0.0, 1.0, 0.0])
        assert list(X.columns) == ["num", "g"]

    def test__a_column_already_called_group_id__does_not_collide(self) -> None:
        X = pd.DataFrame({"group_id": [1.0, 2.0], "g": ["a", "b"]})

        out = GroupIdEncoder("g").fit_transform(X)

        assert list(out.columns) == ["group_id", "group_id_1"]

    def test__declared_categoricals__move_past_the_dropped_columns(self) -> None:
        X = pd.DataFrame({"a": [1], "g": ["x"], "b": [2], "c": [3]})
        encoder = GroupIdEncoder("g")
        encoder.fit_transform(X)

        # 'b' and 'c' each move down one. The selected column itself is gone, so
        # a declaration on it is dropped.
        assert encoder.output_indices([1, 2, 3]) == [1, 2]
        assert encoder.output_indices(None) is None

    def test__mask_unseen_groups__sets_codes_absent_from_training_to_nan(
        self,
    ) -> None:
        X_train = np.array([[5.0, 0.0], [6.0, 1.0], [7.0, np.nan]])
        X_holdout = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, np.nan]])

        masked = mask_unseen_groups(X_train, X_holdout, 1)

        np.testing.assert_array_equal(masked[:, 0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(masked[:, 1], [1.0, np.nan, np.nan])
        assert X_holdout[1, 1] == 2.0


class TestEstimators:
    """Both estimators expose `group_col` and feed the encoded feature to the model."""

    @pytest.mark.parametrize("estimator_cls", ESTIMATORS)
    def test__group_col__survives_a_clone(self, estimator_cls: type) -> None:
        estimator = estimator_cls(group_col=["site", "patient"])

        assert clone(estimator).get_params()["group_col"] == ["site", "patient"]

    @pytest.mark.parametrize("estimator_cls", ESTIMATORS)
    def test__no_group_col__leaves_the_input_as_it_is(
        self, estimator_cls: type
    ) -> None:
        """The default path is unchanged: every column stays an ordinary feature."""
        X, y = _estimator_data(estimator_cls, [f"p{i % 5}" for i in range(30)])
        estimator = estimator_cls(n_estimators=1)

        estimator.fit(X, y)

        assert estimator.n_features_in_ == 3
        assert estimator.inferred_feature_schema_.num_columns == 3
        assert estimator.group_encoder_.feature_name_ is None
        assert estimator.predict(X).shape == (30,)

    @pytest.mark.parametrize("estimator_cls", ESTIMATORS)
    @pytest.mark.parametrize(
        ("group_col", "n_schema_columns"),
        [("g", 3), (["g", "other"], 2)],
        ids=["single", "composite"],
    )
    def test__group_col__replaces_the_columns_with_one_categorical(
        self, estimator_cls: type, group_col: Any, n_schema_columns: int
    ) -> None:
        estimator, X, _ = _fitted(
            estimator_cls, [f"p{i % 7}" for i in range(30)], group_col=group_col
        )

        # The caller still passes the group columns, so the input width is
        # unchanged; the model sees one encoded feature in place of them.
        assert estimator.n_features_in_ == 3
        assert estimator.inferred_feature_schema_.num_columns == n_schema_columns
        index, feature = _group_feature(estimator)
        assert feature.modality is FeatureModality.CATEGORICAL
        flagged = [
            i
            for i, f in enumerate(estimator.inferred_feature_schema_.features)
            if f.is_group_id
        ]
        assert flagged == [index]
        assert estimator.predict(X).shape == (30,)

    @pytest.mark.parametrize("estimator_cls", ESTIMATORS)
    def test__ensemble_members__keep_one_group_feature(
        self, estimator_cls: type
    ) -> None:
        """Through the distribution reshaping and the ordinal encoding."""
        X, y = _estimator_data(estimator_cls, [f"p{i % 4}" for i in range(40)])
        X["colour"] = ["red", "blue"] * 20
        estimator = estimator_cls(n_estimators=4, group_col="g").fit(X, y)

        for member in estimator.executor_.ensemble_members:
            schema = member.cpu_preprocessor.final_feature_schema_
            flagged = [f for f in schema.features if f.is_group_id]
            assert len(flagged) == 1
            assert flagged[0].modality is FeatureModality.CATEGORICAL

    @pytest.mark.parametrize("estimator_cls", ESTIMATORS)
    def test__numeric_group_ids__are_categorical_however_many_there_are(
        self, estimator_cls: type
    ) -> None:
        """One group per row: far past the cardinality cap for a numeric column."""
        estimator, _, _ = _fitted(estimator_cls, list(range(40)), group_col="g")

        _, feature = _group_feature(estimator)
        assert feature.modality is FeatureModality.CATEGORICAL

    @pytest.mark.parametrize("estimator_cls", ESTIMATORS)
    def test__array_input__is_refused_at_fit_and_at_predict(
        self, estimator_cls: type
    ) -> None:
        """`ValueError`, as the `group_col` documentation promises."""
        X, y = _estimator_data(estimator_cls, [f"p{i % 5}" for i in range(30)])
        estimator = estimator_cls(n_estimators=1, group_col="g")

        with pytest.raises(ValueError, match="has to be a DataFrame"):
            estimator.fit(X.to_numpy(), y)

        estimator.fit(X, y)
        # The array is as wide as the fit input, so only the encoder catches it.
        with pytest.raises(ValueError, match="has to be a DataFrame"):
            estimator.predict(X.to_numpy())

    @pytest.mark.parametrize("estimator_cls", ESTIMATORS)
    def test__group_col_that_names_no_column__is_refused_at_fit(
        self, estimator_cls: type
    ) -> None:
        X, y = _estimator_data(estimator_cls, [f"p{i % 5}" for i in range(30)])

        with pytest.raises(ValueError, match="not in the input"):
            estimator_cls(n_estimators=1, group_col="patient").fit(X, y)

    @pytest.mark.parametrize("estimator_cls", ESTIMATORS)
    def test__refit__forgets_the_groups_of_the_first_fit(
        self, estimator_cls: type
    ) -> None:
        estimator, _, _ = _fitted(estimator_cls, ["a", "b"] * 15, group_col="g")

        X, y = _estimator_data(estimator_cls, ["x", "y", "z"] * 10)
        estimator.fit(X, y)

        assert estimator.group_encoder_.keys_.tolist() == [("x",), ("y",), ("z",)]

    @pytest.mark.parametrize("estimator_cls", ESTIMATORS)
    @pytest.mark.parametrize(
        ("groups", "with_string_column"),
        [
            # Few enough distinct codes that detection alone calls them categorical.
            (["a", "b"] * 51, False),
            # Another string column, so cleaning runs its ordinal encoder.
            (["a", "b"] * 51, True),
            # A single group, which detection calls constant.
            (["a"] * 30, False),
        ],
        ids=["detected_categorical", "next_to_a_string_column", "single_group"],
    )
    def test__model_input__carries_the_codes_and_misses_unseen_groups(
        self, estimator_cls: type, groups: list[str], with_string_column: bool
    ) -> None:
        """The codes reach the model untouched by cleaning, unseen keys as NaN."""
        X, y = _estimator_data(estimator_cls, groups)
        if with_string_column:
            X["colour"] = ["red", "blue"] * (len(groups) // 2)
        estimator = estimator_cls(n_estimators=1, group_col="g", fit_mode="low_memory")
        estimator.fit(X, y)

        X_predict = X.iloc[:2].copy()
        X_predict["g"] = ["a", "unseen"]
        seen: list[np.ndarray] = []
        iter_outputs = estimator.executor_.iter_outputs

        def spy(X: np.ndarray, **kwargs: Any) -> Any:
            seen.append(np.asarray(X))
            return iter_outputs(X, **kwargs)

        with patch.object(estimator.executor_, "iter_outputs", spy):
            estimator.predict(X_predict)

        codes = seen[0][:, estimator.group_feature_index_]
        assert codes[0] == 0.0
        assert np.isnan(codes[1])

    @pytest.mark.parametrize("estimator_cls", ESTIMATORS)
    def test__get_embeddings__encodes_the_groups_first(
        self, estimator_cls: type
    ) -> None:
        estimator, X, _ = _fitted(estimator_cls, ["a", "b", "c"] * 10, group_col="g")

        assert estimator.get_embeddings(X).shape[1] == 30

    @pytest.mark.parametrize("estimator_cls", ESTIMATORS)
    def test__tuning_clone__keeps_the_group_feature_categorical(
        self, estimator_cls: type
    ) -> None:
        """The clone fits on the cleaned array, past the numeric cardinality cap."""
        groups = [f"p{i % 40}" for i in range(80)]
        estimator, _, y = _fitted(estimator_cls, groups, group_col="g")
        index = estimator.group_feature_index_
        X_cleaned = np.zeros((80, 3))
        X_cleaned[:, index] = np.arange(80) % 40

        tuning = (
            estimator._get_tuning_classifier()
            if estimator_cls is TabPFNClassifier
            else estimator._get_tuning_regressor()
        )
        tuning.fit(X_cleaned, y)

        feature = tuning.inferred_feature_schema_.features[index]
        assert feature.modality is FeatureModality.CATEGORICAL
        assert feature.is_group_id

    @pytest.mark.parametrize("estimator_cls", ESTIMATORS)
    def test__tuning_fold__sees_holdout_only_groups_as_unseen(
        self, estimator_cls: type
    ) -> None:
        """As a model fitted on the fold's training rows alone would."""
        estimator, _, _ = _fitted(estimator_cls, ["a", "b", "c"] * 10, group_col="g")
        index = estimator.group_feature_index_
        rng = np.random.default_rng(seed=0)
        X_train = rng.normal(size=(20, 3))
        X_train[:, index] = np.arange(20) % 2
        X_holdout = rng.normal(size=(2, 3))
        X_holdout[:, index] = [0.0, 2.0]
        if estimator_cls is TabPFNClassifier:
            y_train, y_holdout = np.arange(20) % 2, np.array([0, 1])
            make_tuning = "_get_tuning_classifier"
        else:
            y_train, y_holdout = rng.normal(size=20), rng.normal(size=2)
            make_tuning = "_get_tuning_regressor"
        split = (X_train, X_holdout, y_train, y_holdout)

        seen: list[np.ndarray] = []
        make = getattr(estimator, make_tuning)

        def make_and_spy(**kwargs: Any) -> Any:
            tuning = make(**kwargs)
            fit = tuning.fit

            def fit_and_spy(X: np.ndarray, y: np.ndarray) -> Any:
                fit(X, y)
                iter_outputs = tuning.executor_.iter_outputs

                def spy(X: np.ndarray, **kwargs: Any) -> Any:
                    seen.append(np.asarray(X))
                    return iter_outputs(X, **kwargs)

                tuning.executor_.iter_outputs = spy
                return tuning

            tuning.fit = fit_and_spy
            return tuning

        module = estimator_cls.__module__
        with (
            patch(f"{module}.get_tuning_splits", return_value=[split]),
            patch.object(estimator, make_tuning, make_and_spy),
        ):
            estimator._compute_holdout_validation_data(
                X=X_train, y=y_train, holdout_frac=0.1, n_folds=1
            )

        codes = seen[0][:, index]
        assert codes[0] == 0.0
        assert np.isnan(codes[1])
