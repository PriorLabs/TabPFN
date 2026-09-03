#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for `TextTransformer`: which columns are text, expanding them, reapplying."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.exceptions import NotFittedError

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.errors import TabPFNValidationError
from tabpfn.inference_config import InferenceConfig
from tabpfn.inference_tuning import ClassifierTuningConfig, RegressorTuningConfig
from tabpfn.preprocessing.datamodel import FeatureModality
from tabpfn.preprocessing.text import _MAX_COLUMNS_IN_WARNING, TextTransformer

#: The default width a text column is expanded to.
N_COMPONENTS = InferenceConfig().TEXTN_COMPONENTS

#: Above the default `MIN_CARDINALITY_FOR_TEXT` of 30, so the column is text.
N_DISTINCT = 40


def _sentences(n: int = N_DISTINCT) -> list[str]:
    return [f"review {i}, a fairly long sentence" for i in range(n)]


def _frame(
    values: list | pd.Series, dtype: str | type | None = "string"
) -> pd.DataFrame:
    """A numeric column beside `values`, a `string` column unless told otherwise."""
    column = values if dtype is None else pd.Series(values, dtype=dtype)
    return pd.DataFrame({"num": np.arange(len(column), dtype=float), "text": column})


def _expander(**kwargs: object) -> TextTransformer:
    return TextTransformer(transform_text=True, **kwargs)  # type: ignore[arg-type]


def _estimator_data(
    estimator_cls: type, text: pd.Series, n: int = 120
) -> tuple[pd.DataFrame, np.ndarray]:
    """A numeric column beside `text`, plus a `y` matching the estimator."""
    rng = np.random.default_rng(seed=42)
    X = pd.DataFrame({"num": rng.normal(size=n), "review": text})
    y = (
        rng.integers(0, 2, size=n)
        if estimator_cls is TabPFNClassifier
        else rng.normal(size=n)
    )
    return X, y


def _review_column(n: int = 120, dtype: str = "string") -> pd.Series:
    return pd.Series(
        [f"review {i}, a fairly long sentence" for i in range(n)], dtype=dtype
    )


class TestSelection:
    """What counts as text: the dtype first, then the distinct-value count.

    Runs on the raw DataFrame, before validation flattens it into one numpy
    array, so the dtype is still visible.
    """

    def test__string_column_above_the_cutoff__is_expanded(self) -> None:
        transformer = _expander()
        transformer.fit_transform(_frame(_sentences()))
        assert transformer.expanded_indices == [1]

    def test__string_column_at_the_cutoff__is_not_expanded(self) -> None:
        X = _frame(_sentences(30))
        transformer = _expander(min_cardinality_for_text=30)
        assert transformer.fit_transform(X) is X
        assert transformer.expanded_indices == []

    def test__object_column__is_never_expanded(self) -> None:
        """An `object` column may hold anything, so it is not read as text (yet)."""
        X = _frame(_sentences(), dtype=object)
        assert X["text"].dtype == object

        transformer = _expander()
        assert transformer.fit_transform(X) is X
        assert transformer.expanded_indices == []

    def test__category_column__is_never_expanded(self) -> None:
        X = _frame(_sentences(), dtype="category")
        assert _expander().fit_transform(X) is X

    def test__declared_categorical_string_column__is_not_expanded(self) -> None:
        X = _frame(_sentences())
        assert _expander(categorical_indices=[1]).fit_transform(X) is X

    def test__flag_off__expands_nothing(self) -> None:
        X = _frame(_sentences())
        transformer = TextTransformer()
        assert transformer.fit_transform(X) is X
        assert transformer.expanded_indices == []
        assert transformer.feature_names_out_ == ["num", "text"]

    def test__non_dataframe_input__is_a_noop(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        transformer = _expander()
        assert transformer.fit_transform(X) is X
        assert transformer.transform(X) is X


class TestExpansion:
    """`TRANSFORM_TEXT` on: a text column becomes numeric features."""

    def test__text_column__becomes_numeric_features(self) -> None:
        X = _frame(_sentences())

        transformer = _expander()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = transformer.fit_transform(X)

        names = transformer.feature_names_out_
        assert out.shape == (N_DISTINCT, 1 + N_COMPONENTS)
        assert list(out.columns) == names
        # The kept columns come first, in order, then the expansion.
        assert names[0] == "num"
        assert all(name.startswith("text_") for name in names[1:])
        assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in out.dtypes)
        assert out.notna().all().all()

    def test__small_vocabulary__yields_fewer_features_without_a_warning(self) -> None:
        """The encoder keeps fewer features than asked when the column has fewer
        n-grams, and skrub warns about it; the count is ours, not the caller's,
        so the warning is not theirs to see either.
        """
        X = _frame(["a" * i for i in range(1, N_DISTINCT + 1)])

        transformer = _expander()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = transformer.fit_transform(X)

        assert 1 < out.shape[1] < 1 + N_COMPONENTS
        assert transformer.transform(X).shape == out.shape

    def test__missing_values__are_encoded(self) -> None:
        X = _frame([*_sentences(), None, None])

        out = _expander().fit_transform(X)

        assert out.shape[0] == N_DISTINCT + 2
        assert out.notna().all().all()

    def test__declared_categorical_indices__are_remapped(self) -> None:
        """The expanded column is gone from where it was, so what came after it
        has moved down.
        """
        X = pd.DataFrame(
            {"text": pd.Series(_sentences(), dtype="string"), "cat": ["a", "b"] * 20}
        )

        transformer = _expander(categorical_indices=[1])
        transformer.fit_transform(X)

        assert transformer.output_indices([1]) == [0]
        assert transformer.output_indices(None) is None
        assert transformer.feature_names_out_[0] == "cat"

    def test__refit_on_a_frame_with_nothing_to_expand__forgets_the_last_fit(
        self,
    ) -> None:
        transformer = _expander()
        transformer.fit_transform(_frame(_sentences()))
        transformer.fit_transform(np.zeros((3, 2)))

        assert transformer.expanded_indices == []
        assert transformer.transform(_frame([1.0, 2.0, 3.0], dtype=None)).shape[1] == 2

    def test__generated_name_colliding_with_an_existing_column__is_deduped(
        self,
    ) -> None:
        first = _expander()
        first.fit_transform(_frame(_sentences()))
        taken = first.feature_names_out_[1]
        X = pd.DataFrame(
            {
                taken: np.zeros(N_DISTINCT),
                "text": pd.Series(_sentences(), dtype="string"),
            }
        )

        transformer = _expander()
        transformer.fit_transform(X)

        names = transformer.feature_names_out_
        assert len(set(names)) == len(names)
        assert names[0] == taken
        assert f"{taken}_1" in names

    def test__duplicate_column_labels__are_handled_positionally(self) -> None:
        """Pandas allows repeated labels, so only position identifies a column."""
        X = pd.DataFrame(
            {0: pd.Series(_sentences(), dtype="string"), 1: np.zeros(N_DISTINCT)}
        )
        X.columns = ["same", "same"]

        transformer = _expander()
        out = transformer.fit_transform(X)

        assert transformer.expanded_indices == [0]
        assert out.shape[1] == 1 + N_COMPONENTS
        assert transformer.feature_names_out_[0] == "same"

    def test__caller_s_frame__is_left_untouched(self) -> None:
        X = _frame(_sentences())
        before = X.copy()

        _expander().fit_transform(X)

        pd.testing.assert_frame_equal(X, before)


class TestExpansionAtPredictTime:
    """`transform` reapplies the encoder fit on the training column, not a new one."""

    def test__same_data__reproduces_the_fitted_columns(self) -> None:
        X = _frame(_sentences())
        transformer = _expander()
        fitted = transformer.fit_transform(X)

        out = transformer.transform(X)

        pd.testing.assert_frame_equal(out, fitted)

    def test__object_column_at_predict__is_read_as_strings(self) -> None:
        """A `string` column at fit arrives as `object` at predict on an older
        pandas, or from a caller who built the frame differently; the values are
        what the encoder reads.
        """
        transformer = _expander()
        fitted = transformer.fit_transform(_frame(_sentences()))

        out = transformer.transform(_frame(_sentences(), dtype=object))

        np.testing.assert_allclose(out.to_numpy(), fitted.to_numpy())

    def test__unseen_values__keep_the_fitted_width(self) -> None:
        transformer = _expander()
        fitted = transformer.fit_transform(_frame(_sentences()))

        out = transformer.transform(_frame(["never seen", None, "", "zzz"] * 10))

        assert list(out.columns) == list(fitted.columns)
        assert out.notna().all().all()

    def test__numeric_column_at_a_text_position__is_refused(self) -> None:
        transformer = _expander()
        transformer.fit_transform(_frame(_sentences()))

        with pytest.raises(TabPFNValidationError, match="hold numbers now") as excinfo:
            transformer.transform(
                _frame(np.arange(N_DISTINCT, dtype=float), dtype=None)
            )
        assert "1 ('text')" in str(excinfo.value)

    def test__array_after_an_expanding_fit__is_refused(self) -> None:
        """Its raw width matches the fit input, so only the transformer can tell
        that it cannot be widened to the expanded layout.
        """
        transformer = _expander()
        transformer.fit_transform(_frame(_sentences()))

        with pytest.raises(TabPFNValidationError, match="has to be a DataFrame"):
            transformer.transform(np.zeros((3, 2)))

    def test__array_after_a_fit_that_expanded_nothing__passes(self) -> None:
        transformer = _expander()
        transformer.fit_transform(_frame([1.0, 2.0, 3.0], dtype=None))

        X = np.zeros((3, 2))
        assert transformer.transform(X) is X


class TestWarning:
    """A text column nothing expands is warned about here, off the raw frame,
    before validation flattens the dtypes away. The model reads it as a number.
    """

    def test__string_column_with_the_flag_off__warns_by_name(self) -> None:
        with pytest.warns(UserWarning, match="look like free text") as record:
            TextTransformer().fit_transform(_frame(_sentences()))

        message = str(record[0].message)
        assert "'text'" in message
        # The message must state every remedy.
        assert "numeric dtype" in message
        assert "categorical_features_indices" in message
        assert "`category` dtype" in message
        assert "MIN_CARDINALITY_FOR_TEXT" in message
        assert "TRANSFORM_TEXT" in message
        assert "https://github.com/PriorLabs/tabpfn-client" in message

    @pytest.mark.parametrize("transform_text", [False, True])
    def test__object_column__warns_whatever_the_flag(
        self, transform_text: bool
    ) -> None:
        X = _frame(_sentences(), dtype=object)

        with pytest.warns(UserWarning, match="look like free text") as record:
            TextTransformer(transform_text=transform_text).fit_transform(X)

        assert "'text'" in str(record[0].message)

    def test__expanded_column__does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _expander().fit_transform(_frame(_sentences()))

    @pytest.mark.parametrize(
        ("column", "declared"),
        [
            (pd.Series(_sentences(), dtype="string"), [1]),
            (pd.Series(_sentences(), dtype="category"), None),
            (pd.Series(_sentences(30), dtype="string"), None),
            (pd.Series([str(i / 7) for i in range(N_DISTINCT)], dtype="string"), None),
            (pd.Series([str(i / 7) for i in range(N_DISTINCT)], dtype=object), None),
            (pd.Series(np.arange(N_DISTINCT, dtype=float)), None),
        ],
        ids=[
            "declared",
            "category_dtype",
            "at_the_cutoff",
            "numeric_strings",
            "numeric_object_strings",
            "numbers",
        ],
    )
    def test__non_text_columns__do_not_warn(
        self, column: pd.Series, declared: list[int] | None
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TextTransformer(categorical_indices=declared).fit_transform(
                _frame(column, dtype=None)
            )

    def test__numeric_column_with_one_stray_token__warns(self) -> None:
        """One non-numeric token makes a numeric column text: every value has to
        parse for it to be a number. The fix the warning asks for is a numeric
        dtype.
        """
        values = np.random.default_rng(2).normal(size=200)
        mostly_numeric = [str(round(float(v), 4)) for v in values]
        mostly_numeric[7] = "N/A"

        with pytest.warns(UserWarning, match="look like free text") as record:
            TextTransformer().fit_transform(_frame(mostly_numeric, dtype=object))

        assert "'text'" in str(record[0].message)

    def test__column_with_a_crash_prone_token__does_not_crash(self) -> None:
        """A value that used to segfault `pandas.to_numeric` must not crash the fit.

        `"8e2569614270f3d8b9e7038efac9f116"` reads as scientific notation with an
        exponent in `[2**31, 2**32)`, which crashes `pandas.to_numeric` outright on
        some pandas/numpy versions. Surviving the call is the assertion.
        """
        values = [f"id_{i}" for i in range(200)]
        values[7] = "8e2569614270f3d8b9e7038efac9f116"

        with pytest.warns(UserWarning, match="look like free text"):
            TextTransformer().fit_transform(_frame(values, dtype=object))

    def test__many_text_columns__message_is_truncated(self) -> None:
        n_extra = 5
        n_columns = _MAX_COLUMNS_IN_WARNING + n_extra
        X = pd.DataFrame(
            {f"t{i}": pd.Series(_sentences(), dtype="string") for i in range(n_columns)}
        )

        with pytest.warns(UserWarning, match="look like free text") as record:
            TextTransformer().fit_transform(X)

        message = str(record[0].message)
        assert f"(and {n_extra} more)" in message
        assert f"'t{_MAX_COLUMNS_IN_WARNING - 1}'" in message
        assert f"'t{_MAX_COLUMNS_IN_WARNING}'" not in message

    def test__transform__does_not_warn_again(self) -> None:
        X = _frame(_sentences())
        transformer = TextTransformer()
        with pytest.warns(UserWarning, match="look like free text"):
            transformer.fit_transform(X)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            transformer.transform(X)


class TestInterface:
    """`fit`, `fit_transform` and `transform` relate the way sklearn's do."""

    def test__fit__returns_itself_and_transform_matches_fit_transform(self) -> None:
        X = _frame(_sentences())

        fitted = _expander().fit(X)
        via_transform = fitted.transform(X)
        via_fit_transform = _expander().fit_transform(X)

        assert isinstance(fitted, TextTransformer)
        pd.testing.assert_frame_equal(via_transform, via_fit_transform)

    def test__transform__before_fit__raises(self) -> None:
        with pytest.raises(NotFittedError):
            TextTransformer().transform(_frame([1.0, 2.0, 3.0], dtype=None))

    def test__expanded_indices__before_fit__raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = TextTransformer().expanded_indices

    def test__fit_on_an_array__has_no_output_names(self) -> None:
        transformer = _expander().fit(np.zeros((3, 2)))

        assert transformer.feature_names_out_ is None
        assert transformer.output_indices([0, 1]) == [0, 1]


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_transform_text__expands_the_text_column(estimator_cls: type) -> None:
    """`TRANSFORM_TEXT` reaches the transformer, and the wider frame survives the
    whole fit/predict path: the schema describes the text features, and the
    free-text warning has nothing left to report.
    """
    X, y = _estimator_data(estimator_cls, _review_column())

    model = estimator_cls(
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_TEXT": True}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model.fit(X, y)
    predictions = model.predict(X)

    schema = model.inferred_feature_schema_
    names = [feature.name for feature in schema.features]
    assert len(names) == 1 + N_COMPONENTS
    assert sum(name.split("input_")[-1].startswith("review_") for name in names) == (
        N_COMPONENTS
    )
    assert schema.indices_for(FeatureModality.CATEGORICAL) == []
    assert len(predictions) == len(X)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_text_n_components__sets_the_expanded_width(
    estimator_cls: type,
) -> None:
    X, y = _estimator_data(estimator_cls, _review_column())

    model = estimator_cls(
        n_estimators=1,
        device="cpu",
        inference_config={"TRANSFORM_TEXT": True, "TEXT_N_COMPONENTS": 5},
    ).fit(X, y)

    assert len(model.inferred_feature_schema_.features) == 1 + 5


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_transform_text__reports_the_caller_s_own_columns(
    estimator_cls: type,
) -> None:
    """Expansion widens the frame internally, which must not leak into the
    sklearn attributes: they describe what the caller passed.
    """
    X, y = _estimator_data(estimator_cls, _review_column())

    model = estimator_cls(
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_TEXT": True}
    ).fit(X, y)

    assert model.n_features_in_ == 2
    assert list(model.feature_names_in_) == ["num", "review"]
    assert len(model.inferred_feature_schema_.features) > 2
    # The right width, but no columns to expand.
    with pytest.raises(TabPFNValidationError, match="has to be a DataFrame"):
        model.predict(np.zeros((10, 2)))


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_text_column__warns_at_call_site(estimator_cls: type) -> None:
    """`fit` runs the transformer before validation, so a text column warns from
    there, blaming this file's `fit` call (the stacklevel); declaring the column
    in `categorical_features_indices` silences it, and `predict` stays quiet.
    """
    X, y = _estimator_data(estimator_cls, _review_column())

    model = estimator_cls(n_estimators=1, device="cpu")
    with pytest.warns(UserWarning, match="look like free text") as record:
        model.fit(X, y)
    assert "'review'" in str(record[0].message)
    assert record[0].filename == __file__

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.predict(X)
    assert not [w for w in caught if "look like free text" in str(w.message)]

    model = estimator_cls(
        n_estimators=1, device="cpu", categorical_features_indices=[1]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
    assert not [w for w in caught if "look like free text" in str(w.message)]


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_without_transform_text__warns_and_keeps_the_width(
    estimator_cls: type,
) -> None:
    """Off by default: the column is read as before, a high-cardinality category
    the fit warns about.
    """
    X, y = _estimator_data(estimator_cls, _review_column())

    model = estimator_cls(n_estimators=1, device="cpu")
    with pytest.warns(UserWarning, match="look like free text") as record:
        model.fit(X, y)

    assert "TRANSFORM_TEXT" in str(record[0].message)
    assert model.inferred_feature_schema_.indices_for(FeatureModality.NUMERICAL) == [
        0,
        1,
    ]


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_transform_text_and_an_object_column__does_not_expand_it(
    estimator_cls: type,
) -> None:
    """Only a `string` dtype is text; an `object` column is read as before."""
    X, y = _estimator_data(estimator_cls, _review_column(dtype="object"))
    assert X["review"].dtype == object

    model = estimator_cls(
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_TEXT": True}
    )
    with pytest.warns(UserWarning, match="look like free text"):
        model.fit(X, y)

    assert model.text_transformer_.expanded_indices == []
    assert model.inferred_feature_schema_.indices_for(FeatureModality.NUMERICAL) == [
        0,
        1,
    ]


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
@pytest.mark.parametrize("transform_text", [False, True])
def test__fit_with_a_category_column_above_the_text_cutoff__reads_it_as_categorical(
    estimator_cls: type, transform_text: bool
) -> None:
    """An explicit `category` dtype is a declaration: neither text, whatever the
    flag, nor anything to warn about. Read off the frame, where the dtype still
    exists; validation flattens it away before detection runs.
    """
    n = 200
    X, y = _estimator_data(
        estimator_cls,
        pd.Series([f"sku_{i % 60}" for i in range(n)], dtype="category"),
        n,
    )

    model = estimator_cls(
        n_estimators=1,
        device="cpu",
        inference_config={"TRANSFORM_TEXT": transform_text},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model.fit(X, y)

    assert model.text_transformer_.expanded_indices == []
    schema = model.inferred_feature_schema_
    assert schema.indices_for(FeatureModality.CATEGORICAL) == [1]
    assert len(model.predict(X)) == n


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_a_declared_categorical_string_column__reads_it_as_categorical(
    estimator_cls: type,
) -> None:
    """A position in `categorical_features_indices` settles it the same way,
    at any cardinality, so the column is neither expanded nor warned about.
    """
    n = 200
    X, y = _estimator_data(
        estimator_cls, pd.Series([f"sku_{i % 60}" for i in range(n)], dtype="string"), n
    )

    model = estimator_cls(
        n_estimators=1,
        device="cpu",
        categorical_features_indices=[1],
        inference_config={"TRANSFORM_TEXT": True},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model.fit(X, y)

    assert model.text_transformer_.expanded_indices == []
    assert model.inferred_feature_schema_.indices_for(FeatureModality.CATEGORICAL) == [
        1
    ]


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_differentiable_input__sets_a_text_transformer(
    estimator_cls: type,
) -> None:
    """The differentiable path fits on a tensor, which holds no strings, and still
    sets the attribute: every predict path converts through it unconditionally.
    """
    X = torch.randn(20, 3)
    y = (
        torch.tensor([0, 1] * 10)
        if estimator_cls is TabPFNClassifier
        else torch.randn(20)
    )
    model = estimator_cls(
        n_estimators=1,
        device="cpu",
        differentiable_input=True,
        ignore_pretraining_limits=True,
    ).fit_with_differentiable_input(X, y)

    assert model.text_transformer_.transform(np.zeros((2, 3))).shape == (2, 3)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_transform_text_and_tuning__tuning_estimator_gets_shifted_indices(
    estimator_cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tuning estimators are fit on the expanded array, where a declared
    categorical column has moved down past the expanded text column, so they
    must be handed the shifted index rather than the caller's.
    """
    n = 80
    rng = np.random.default_rng(seed=0)
    X = pd.DataFrame(
        {
            "review": pd.Series(_review_column(n), dtype="string"),
            "cat": rng.choice(["a", "b", "c"], size=n),
            "num": rng.normal(size=n),
        }
    )
    is_classifier = estimator_cls is TabPFNClassifier
    y = rng.integers(0, 2, size=n) if is_classifier else rng.normal(size=n)
    config_cls = ClassifierTuningConfig if is_classifier else RegressorTuningConfig
    model = estimator_cls(
        n_estimators=1,
        device="cpu",
        categorical_features_indices=[1],
        inference_config={"TRANSFORM_TEXT": True},
        tuning_config=config_cls(
            calibrate_temperature=True, tuning_holdout_frac=0.25, tuning_n_folds=1
        ),
    )
    getter = "_get_tuning_classifier" if is_classifier else "_get_tuning_regressor"
    tuning_estimators: list[TabPFNClassifier | TabPFNRegressor] = []
    original = getattr(model, getter)

    def capture(**kwargs: object) -> TabPFNClassifier | TabPFNRegressor:
        tuning_estimators.append(original(**kwargs))
        return tuning_estimators[-1]

    monkeypatch.setattr(model, getter, capture)
    model.fit(X, y)

    assert tuning_estimators
    assert all(
        estimator.categorical_features_indices == [0] for estimator in tuning_estimators
    )


def test__predict_proba_batched_with_transform_text__expands_the_test_frames() -> None:
    """The batched path validates each test frame the way `predict` does, so an
    expanded text column has to be expanded there too, or the widths disagree.
    """
    X, y = _estimator_data(TabPFNClassifier, _review_column())

    model = TabPFNClassifier(
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_TEXT": True}
    )
    probabilities = model.predict_proba_batched([X, X], [y, y], [X, X])

    assert probabilities.shape[:2] == (2, len(X))
