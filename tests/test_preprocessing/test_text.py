#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for `TextTransformer`: which columns are text, expanding them, reapplying."""

from __future__ import annotations

import warnings
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.exceptions import NotFittedError

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.errors import TabPFNValidationError
from tabpfn.inference_config import InferenceConfig
from tabpfn.inference_tuning import ClassifierTuningConfig, RegressorTuningConfig
from tabpfn.preprocessing.datamodel import INPUT_FEATURE_PREFIX, FeatureModality
from tabpfn.preprocessing.text import TextTransformer

#: The default width a text column is expanded to.
N_COMPONENTS = InferenceConfig.TEXT_N_COMPONENTS

#: Above the default `MIN_CARDINALITY_FOR_TEXT` of 30, so the column is text.
N_DISTINCT = 40


def _sentences(n: int = N_DISTINCT) -> list[str]:
    return [f"review {i}, a fairly long sentence" for i in range(n)]


def _frame(values: list | pd.Series, dtype: object = "string") -> pd.DataFrame:
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


def _pyarrow() -> ModuleType:
    # pyarrow is deliberately not a dependency, not even of the tests, so the
    # tests that need it run wherever it happens to be installed and skip
    # elsewhere, CI included.
    return pytest.importorskip("pyarrow")


def _captured_tuning_estimators(
    model: TabPFNClassifier | TabPFNRegressor, monkeypatch: pytest.MonkeyPatch
) -> list[TabPFNClassifier | TabPFNRegressor]:
    """Every tuning estimator `model.fit` builds, collected as it is built."""
    is_classifier = isinstance(model, TabPFNClassifier)
    getter = "_get_tuning_classifier" if is_classifier else "_get_tuning_regressor"
    captured: list[TabPFNClassifier | TabPFNRegressor] = []
    original = getattr(model, getter)

    def capture(**kwargs: object) -> TabPFNClassifier | TabPFNRegressor:
        captured.append(original(**kwargs))
        return captured[-1]

    monkeypatch.setattr(model, getter, capture)
    return captured


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

    def test__numeric_strings__are_not_expanded(self) -> None:
        """Numbers stored as strings are numbers, not text."""
        X = _frame([str(i / 7) for i in range(N_DISTINCT)])
        assert _expander().fit_transform(X) is X

    def test__pyarrow_string_column__is_expanded(self) -> None:
        """`read_csv(dtype_backend="pyarrow")` and parquet readers hand out this
        dtype, a sibling of `string` rather than a storage of it.
        """
        pa = _pyarrow()
        X = _frame(_sentences(), dtype=pd.ArrowDtype(pa.string()))

        transformer = _expander()
        out = transformer.fit_transform(X)

        assert transformer.expanded_indices == [1]
        assert out.shape[1] == 1 + N_COMPONENTS

    def test__pyarrow_numeric_strings__are_not_expanded(self) -> None:
        pa = _pyarrow()
        X = _frame(
            [str(i / 7) for i in range(N_DISTINCT)], dtype=pd.ArrowDtype(pa.string())
        )
        assert _expander().fit_transform(X) is X

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

    def test__pyarrow_string_column_at_predict__is_read_as_strings(self) -> None:
        pa = _pyarrow()
        transformer = _expander()
        fitted = transformer.fit_transform(_frame(_sentences()))

        out = transformer.transform(
            _frame(_sentences(), dtype=pd.ArrowDtype(pa.string()))
        )

        np.testing.assert_allclose(out.to_numpy(), fitted.to_numpy())

    def test__unseen_values__keep_the_fitted_width(self) -> None:
        transformer = _expander()
        fitted = transformer.fit_transform(_frame(_sentences()))

        out = transformer.transform(_frame(["never seen", None, "", "zzz"] * 10))

        assert list(out.columns) == list(fitted.columns)
        assert out.notna().all().all()

    def test__all_missing_column_at_predict__is_read_as_missing_strings(self) -> None:
        """A column with no values arrives as float, which is how pandas reads an
        empty column, not as a column of numbers: its rows are encoded like any
        missing value.
        """
        transformer = _expander()
        fitted = transformer.fit_transform(_frame(_sentences()))

        out = transformer.transform(_frame([np.nan] * 3, dtype=None))

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
    assert schema.indices_for(FeatureModality.TEXT) == []
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
def test__predict_with_an_all_missing_text_column__is_not_refused(
    estimator_cls: type,
) -> None:
    """A predict frame whose text column is entirely missing, as `read_csv` or a
    `.loc` slice can produce, holds no numbers and is scored, not refused.
    """
    X, y = _estimator_data(estimator_cls, _review_column())
    model = estimator_cls(
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_TEXT": True}
    ).fit(X, y)
    X_predict = X.head(5).copy()
    X_predict["review"] = np.nan
    assert X_predict["review"].dtype == float

    assert len(model.predict(X_predict)) == 5


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
def test__fit_without_transform_text__reads_the_column_as_before(
    estimator_cls: type,
) -> None:
    """Off by default: the column is untouched, so detection labels it text and
    warns about it, as without this transformer.
    """
    X, y = _estimator_data(estimator_cls, _review_column())

    model = estimator_cls(n_estimators=1, device="cpu")
    with pytest.warns(UserWarning, match="look like free text") as record:
        model.fit(X, y)

    assert "'review'" in str(record[0].message)
    assert "TRANSFORM_TEXT" in str(record[0].message)
    assert model.text_transformer_.expanded_indices == []
    assert model.inferred_feature_schema_.indices_for(FeatureModality.TEXT) == [1]


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
@pytest.mark.parametrize("dtype", ["object", "category"])
def test__fit_with_transform_text_and_another_dtype__does_not_expand_it(
    estimator_cls: type, dtype: str
) -> None:
    """Only a `string` dtype is text: an `object` or `category` column is read
    as before, whatever the flag.
    """
    X, y = _estimator_data(estimator_cls, _review_column(dtype=dtype))
    assert not isinstance(X["review"].dtype, pd.StringDtype)

    model = estimator_cls(
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_TEXT": True}
    )
    with pytest.warns(UserWarning, match="look like free text"):
        model.fit(X, y)

    assert model.text_transformer_.expanded_indices == []
    assert len(model.inferred_feature_schema_.features) == 2


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_transform_text_and_a_declared_categorical__does_not_expand_it(
    estimator_cls: type,
) -> None:
    """A position in `categorical_features_indices` is never text, at any
    cardinality: the column is read as before, and nothing warns about it.
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
    assert len(model.inferred_feature_schema_.features) == 2


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
def test__fit_with_transform_text__stores_the_shifted_categorical_indices(
    estimator_cls: type,
) -> None:
    """The fitted attribute addresses the validated input, where the declared
    column has moved down past the expanded text column.
    """
    n = 80
    rng = np.random.default_rng(seed=0)
    X = pd.DataFrame(
        {
            "review": _review_column(n),
            "cat": rng.choice(["a", "b", "c"], size=n),
            "num": rng.normal(size=n),
        }
    )
    y = rng.integers(0, 2, size=n) if estimator_cls is TabPFNClassifier else X["num"]

    model = estimator_cls(
        n_estimators=1,
        device="cpu",
        categorical_features_indices=[1],
        inference_config={"TRANSFORM_TEXT": True},
    ).fit(X, y)
    assert model.categorical_features_indices_ == [0]

    model = estimator_cls(
        n_estimators=1, device="cpu", inference_config={"TRANSFORM_TEXT": True}
    ).fit(X, y)
    assert model.categorical_features_indices_ is None


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
            "review": _review_column(n),
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
    tuning_estimators = _captured_tuning_estimators(model, monkeypatch)
    model.fit(X, y)

    assert tuning_estimators
    assert all(
        estimator.categorical_features_indices == [0] for estimator in tuning_estimators
    )


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_dates_and_text__declared_categorical_moves_past_both(
    estimator_cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared categorical column behind an expanded date and an expanded text
    column moves down once per expansion, since both append their features at
    the end. The stored indices, the schema and the tuning estimators must all
    find it where it ends up.
    """
    n = 80
    rng = np.random.default_rng(seed=0)
    X = pd.DataFrame(
        {
            "when": pd.date_range("2021-01-01", periods=n, freq="D"),
            "review": _review_column(n),
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
        categorical_features_indices=[2],
        inference_config={"TRANSFORM_DATES": True, "TRANSFORM_TEXT": True},
        tuning_config=config_cls(
            calibrate_temperature=True, tuning_holdout_frac=0.25, tuning_n_folds=1
        ),
    )
    tuning_estimators = _captured_tuning_estimators(model, monkeypatch)
    model.fit(X, y)

    schema = model.inferred_feature_schema_
    names = [
        feature.name.removeprefix(INPUT_FEATURE_PREFIX) for feature in schema.features
    ]
    assert names[:2] == ["cat", "num"]
    assert all(name.startswith(("when_", "review_")) for name in names[2:])
    assert model.date_transformer_.expanded_indices == [0]
    # `review` sat at 1 and moved down once the date column ahead of it was dropped.
    assert model.text_transformer_.expanded_indices == [0]
    assert model.categorical_features_indices_ == [0]
    assert schema.indices_for(FeatureModality.CATEGORICAL) == [0]
    assert tuning_estimators
    assert all(
        estimator.categorical_features_indices == [0] for estimator in tuning_estimators
    )
    assert len(model.predict(X)) == n


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
