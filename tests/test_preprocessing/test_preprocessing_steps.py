#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import inspect
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING
from typing_extensions import override
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import sklearn
import sklearn.base
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    OneHotEncoder,
    OrdinalEncoder,
)

from tabpfn.preprocessing import steps
from tabpfn.preprocessing.datamodel import Feature, FeatureModality, FeatureSchema
from tabpfn.preprocessing.pipeline_interface import (
    PreprocessingPipeline,
    PreprocessingStep,
)
from tabpfn.preprocessing.steps import (
    AddFingerprintFeaturesStep,
    AddSVDFeaturesStep,
    DifferentiableZNormStep,
    ReshapeFeatureDistributionsStep,
)
from tabpfn.preprocessing.steps.preprocessing_helpers import (
    EfficientColumnTransformer,
    OrderPreservingColumnTransformer,
    get_ordinal_encoder,
)
from tabpfn.preprocessing.steps.remove_constant_features_step import (
    RemoveConstantFeaturesStep,
)
from tabpfn.preprocessing.steps.utils import is_identity_transformer

if TYPE_CHECKING:
    from tabpfn.classifier import XType, YType


def _get_schema(num_columns: int) -> FeatureSchema:
    """Create a schema with all numerical features."""
    return FeatureSchema(
        features=[
            Feature(name=f"f{i}", modality=FeatureModality.NUMERICAL)
            for i in range(num_columns)
        ]
    )


def _get_preprocessing_steps() -> list[Callable[..., PreprocessingStep],]:
    defaults: list[Callable[..., PreprocessingStep]] = [
        cls
        for cls in steps.__dict__.values()
        if (
            isinstance(cls, type)
            and issubclass(cls, PreprocessingStep)
            and cls is not PreprocessingStep
            and cls is not DifferentiableZNormStep  # works on torch tensors
        )
    ]
    extras: list[Callable[..., PreprocessingStep]] = [
        partial(
            ReshapeFeatureDistributionsStep,
            transform_name="none",
            append_to_original="auto",
            apply_to_categorical=False,
        )
    ]
    return defaults + extras


def _get_random_data(
    rng: np.random.Generator, n_samples: int, n_features: int, cat_inds: list[int]
) -> np.ndarray:
    x = rng.random((n_samples, n_features))
    x[:, cat_inds] = rng.integers(0, 3, size=(n_samples, len(cat_inds))).astype(float)
    return x


def _make_metadata(n_features: int, cat_inds: list[int]) -> FeatureSchema:
    return FeatureSchema.from_only_categorical_indices(cat_inds, n_features)


def test__preprocessing_steps__transform__is_idempotent():
    """Test that calling transform multiple times on the same data
    gives the same result. This ensures transform is deterministic
    and doesn't have internal state changes.
    """
    rng = np.random.default_rng(42)
    n_samples = 20
    n_features = 4
    cat_inds = [1, 3]
    feature_schema = _make_metadata(n_features, cat_inds)
    for cls in _get_preprocessing_steps():
        x = _get_random_data(rng, n_samples, n_features, cat_inds)
        x2 = _get_random_data(rng, n_samples, n_features, cat_inds)

        obj = cls()
        obj.fit_transform(x, feature_schema)

        # Calling transform multiple times should give the same result
        result1 = obj.transform(x2)
        result2 = obj.transform(x2)

        assert np.allclose(result1.X, result2.X), f"Transform not idempotent for {cls}"
        assert result1.feature_schema.indices_for(
            FeatureModality.CATEGORICAL
        ) == result2.feature_schema.indices_for(FeatureModality.CATEGORICAL)


def test__preprocessing_steps__transform__no_sample_interdependence():
    """Test that preprocessing steps don't have
    interdependence between samples during transform. Each sample should be
    transformed independently based only on parameters learned during fit.
    """
    rng = np.random.default_rng(42)
    n_samples = 20
    n_features = 4
    cat_inds = [1, 3]
    feature_schema = _make_metadata(n_features, cat_inds)
    for cls in _get_preprocessing_steps():
        x = _get_random_data(rng, n_samples, n_features, cat_inds)
        x2 = _get_random_data(rng, n_samples, n_features, cat_inds)

        obj = cls()
        obj.fit_transform(x, feature_schema)

        # Test 1: Shuffling samples should give correspondingly shuffled results
        result_normal = obj.transform(x2)
        result_reversed = obj.transform(x2[::-1])
        assert np.allclose(result_reversed.X[::-1], result_normal.X), (
            f"Transform depends on sample order for {cls}"
        )

        # Test 2: Transforming a subset should match the subset of full transformation
        result_full = obj.transform(x2)
        result_subset = obj.transform(x2[:4])
        assert np.allclose(result_full.X[:4], result_subset.X), (
            f"Transform depends on other samples in batch for {cls}"
        )

        # Test 3: Categorical features should remain the same
        assert result_full.feature_schema.indices_for(
            FeatureModality.CATEGORICAL
        ) == result_subset.feature_schema.indices_for(FeatureModality.CATEGORICAL)


def _make_step(cls: Callable[..., PreprocessingStep]) -> PreprocessingStep:
    """Create a step, pinning random_state=0 when the constructor accepts it."""
    sig = inspect.signature(cls)
    if "random_state" in sig.parameters:
        return cls(random_state=0)
    return cls()


def test__preprocessing_steps__refit_safety():
    """Fitting a step on dataset A then re-fitting on dataset B must produce
    the same result as a fresh step fit only on dataset B.

    This guards against internal state (e.g. cached flags, overwritten init
    params) leaking across fits.
    """
    rng = np.random.default_rng(42)
    cat_inds = [1, 3]

    # Dataset A: small, few features.
    n_a, f_a = 30, 4
    schema_a = _make_metadata(f_a, cat_inds)
    x_a = _get_random_data(rng, n_a, f_a, cat_inds)

    # Dataset B: larger, more features (triggers different code-paths in
    # steps that branch on feature count, e.g. SVD, append_to_original).
    n_b, f_b = 50, 8
    cat_inds_b = [1, 3]
    schema_b = _make_metadata(f_b, cat_inds_b)
    x_b = _get_random_data(rng, n_b, f_b, cat_inds_b)

    for cls in _get_preprocessing_steps():
        # Fit on A, then re-fit on B with the same instance.
        step_reused = _make_step(cls)
        step_reused.fit_transform(x_a, schema_a)
        reused_result = step_reused.fit_transform(x_b, schema_b)

        # Fresh instance fit only on B.
        step_fresh = _make_step(cls)
        fresh_result = step_fresh.fit_transform(x_b, schema_b)

        assert reused_result.X.shape == fresh_result.X.shape, (
            f"Refit shape mismatch for {cls}: "
            f"{reused_result.X.shape} vs {fresh_result.X.shape}"
        )
        assert np.allclose(reused_result.X, fresh_result.X, equal_nan=True), (
            f"Refit output mismatch for {cls}"
        )
        if reused_result.X_added is not None or fresh_result.X_added is not None:
            assert reused_result.X_added is not None, (
                f"Refit X_added presence mismatch for {cls}: reused_result.X_added is None"  # noqa: E501
            )
            assert fresh_result.X_added is not None, (
                f"Refit X_added presence mismatch for {cls}: fresh_result.X_added is None"  # noqa: E501
            )

            assert np.allclose(
                reused_result.X_added, fresh_result.X_added, equal_nan=True
            ), f"Refit X_added mismatch for {cls}"


def test__pipeline__handles_added_columns_from_fingerprint_step():
    """Test that the pipeline correctly handles added_columns from steps.

    The fingerprint step returns X unchanged and provides the fingerprint
    via added_columns. The pipeline should concatenate this and update schema.
    """
    rng = np.random.default_rng(42)
    n_samples, n_features = 10, 3
    X = rng.random((n_samples, n_features))
    schema = FeatureSchema(
        features=[
            Feature(name=f"f{i}", modality=FeatureModality.NUMERICAL)
            for i in range(n_features)
        ]
    )

    # Create pipeline with fingerprint step
    fingerprint_step = AddFingerprintFeaturesStep()
    pipeline = PreprocessingPipeline(steps=[fingerprint_step])

    result = pipeline.fit_transform(X, schema)

    # Pipeline should have concatenated the fingerprint column
    assert result.X.shape == (n_samples, n_features + 1)

    # Metadata should track the new column
    assert result.feature_schema.num_columns == n_features + 1
    assert (
        len(result.feature_schema.indices_for(FeatureModality.NUMERICAL))
        == n_features + 1
    )

    # Original columns should be preserved
    np.testing.assert_array_equal(result.X[:, :n_features], X)


def test__pipeline__transform_also_handles_added_columns():
    """Test that pipeline.transform also correctly handles added_columns."""
    rng = np.random.default_rng(42)
    n_samples, n_features = 10, 3
    X_train = rng.random((n_samples, n_features))
    X_test = rng.random((5, n_features))
    schema = FeatureSchema(
        features=[
            Feature(name=f"f{i}", modality=FeatureModality.NUMERICAL)
            for i in range(n_features)
        ]
    )

    # Create and fit pipeline
    fingerprint_step = AddFingerprintFeaturesStep()
    pipeline = PreprocessingPipeline(steps=[fingerprint_step])
    pipeline.fit_transform(X_train, schema)

    # Transform test data
    result = pipeline.transform(X_test)

    # Should also have the fingerprint column
    assert result.X.shape == (5, n_features + 1)


# TODO: Ideally we don't allow for this in no preprocessing step!
def test__pipeline__raises_error_when_modality_step_changes_column_count():
    """Test that pipeline raises error if modality-registered step changes columns."""

    class BadStep(PreprocessingStep):
        """A step that incorrectly returns more columns than it received."""

        @override
        def _fit(self, X: np.ndarray, metadata: FeatureSchema) -> FeatureSchema:
            return metadata

        @override
        def _transform(
            self, X: np.ndarray, *, is_test: bool = False
        ) -> tuple[np.ndarray, None, None]:
            # Incorrectly return more columns
            return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1), None, None

    rng = np.random.default_rng(42)
    X = rng.random((10, 3))
    schema = FeatureSchema(
        features=[
            Feature(name=f"f{i}", modality=FeatureModality.NUMERICAL) for i in range(3)
        ]
    )

    # Register step with modalities - should raise error
    bad_step = BadStep()
    pipeline = PreprocessingPipeline(steps=[(bad_step, {FeatureModality.NUMERICAL})])

    with pytest.raises(ValueError, match="received 3 columns but returned 4"):
        pipeline.fit_transform(X, schema)


def test__order_preserving_column_transformer():
    """Should raise ValueError if column sets overlap."""
    ordinal_enc1 = OrdinalEncoder()
    ordinal_enc2 = OrdinalEncoder()
    onehotencoder1 = OneHotEncoder()

    # --- Mock dataset ---
    mock_data_df = pd.DataFrame(
        {
            "a": [10, 20, 30, 40],
            "b": ["x", "y", "x", "z"],
        }
    )

    # Test error raised due to too many transformers
    multiple_transformers = [
        ("ordinal_enc1", ordinal_enc1, ["a"]),
        ("ordinal_enc2", ordinal_enc2, ["b"]),
    ]

    with pytest.raises(
        ValueError,
        match="OrderPreservingColumnTransformer only supports up to one transformer",
    ):
        OrderPreservingColumnTransformer(transformers=multiple_transformers).fit(
            mock_data_df
        )

    # Test error, due to unsupported encoder type (OneHotEncoder)
    incompatible_transformer = [("onehot", onehotencoder1, ["b"])]

    with pytest.raises(ValueError, match="are instances of OneToOneFeatureMixin"):
        OrderPreservingColumnTransformer(transformers=incompatible_transformer).fit(
            mock_data_df
        )

    # Test if normal column transformer shuffles column order,
    # while the OrderPreserving restores the original order
    non_overlapping_ordinal_encoder = [("ordinal_enc1", ordinal_enc1, ["b"])]

    vanilla_transformer = ColumnTransformer(
        transformers=non_overlapping_ordinal_encoder, remainder=FunctionTransformer()
    )

    vanilla_output = vanilla_transformer.fit_transform(mock_data_df)

    # Vanilla transformer shuffles column order
    assert not np.array_equal(mock_data_df.iloc[:, 0].values, vanilla_output[:, 0])

    preserving_transformer = OrderPreservingColumnTransformer(
        transformers=non_overlapping_ordinal_encoder, remainder=FunctionTransformer()
    )

    # OrderPreserving transformer does not shuffle column order
    preserved_output = preserving_transformer.fit_transform(mock_data_df)
    np.testing.assert_equal(mock_data_df.iloc[:, 0].values, preserved_output[:, 0])


def test__order_preserving_column_transformer__clone_keeps_its_parameters() -> None:
    """Every parameter it was configured with has to survive sklearn's `clone`."""
    encoder = get_ordinal_encoder()

    assert set(encoder.get_params(deep=False)) == set(
        ColumnTransformer(transformers=[]).get_params(deep=False)
    )

    copy = sklearn.base.clone(encoder)

    # A `remainder` lost to its 'drop' default silently drops every numeric column
    assert isinstance(copy.remainder, FunctionTransformer)
    assert copy.sparse_threshold == encoder.sparse_threshold
    assert copy.verbose_feature_names_out == encoder.verbose_feature_names_out


@pytest.mark.parametrize("columns", [slice(0, 2), "b", 1, ("a", "b")])
def test__order_preserving_column_transformer__selection_not_a_list_of_keys(
    columns: object,
) -> None:
    """A selection that cannot be placed one column at a time is refused."""
    X = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    transformer = OrderPreservingColumnTransformer(
        transformers=[("encoder", OrdinalEncoder(), columns)]
    )

    with pytest.raises(ValueError, match="only supports selecting columns by a"):
        transformer.fit(X)


@pytest.mark.parametrize("remainder", ["drop", OneHotEncoder()])
def test__order_preserving_column_transformer__remainder_that_drops_columns(
    remainder: object,
) -> None:
    """A column the remainder never hands back has nowhere to go in the input's order.

    'drop' is `ColumnTransformer`'s default, so this is the shape reached by leaving
    the parameter off entirely.
    """
    X = pd.DataFrame({"a": [1.0, 2.0], "b": ["p", "q"]})
    transformer = OrderPreservingColumnTransformer(
        transformers=[("encoder", OrdinalEncoder(), ["b"])],
        remainder=remainder,
        sparse_threshold=0.0,
    )

    with pytest.raises(ValueError, match="a remainder that hands every column"):
        transformer.fit_transform(X)


@pytest.mark.parametrize(
    "transformers",
    [
        [("encoder", OrdinalEncoder(), ["a"]), ("second", OrdinalEncoder(), ["b"])],
        [("encoder", OneHotEncoder(), ["a"])],
        [("encoder", OrdinalEncoder(), slice(0, 1))],
    ],
    ids=["two_transformers", "not_one_to_one", "selection_not_keys"],
)
def test__order_preserving_column_transformer__contract_survives_set_params(
    transformers: list,
) -> None:
    """`set_params` writes past the constructor, so fit is what has to hold the line."""
    X = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    transformer = OrderPreservingColumnTransformer(
        transformers=[("encoder", OrdinalEncoder(), ["a"])],
        remainder=FunctionTransformer(),
        sparse_threshold=0.0,
    )

    transformer.set_params(transformers=transformers)

    with pytest.raises(ValueError, match="OrderPreservingColumnTransformer only"):
        transformer.fit_transform(X)


def test__order_preserving_column_transformer__callable_selects_a_slice() -> None:
    """What the constructor cannot see, the reorder says outright rather than crash."""
    X = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    transformer = OrderPreservingColumnTransformer(
        transformers=[("encoder", OrdinalEncoder(), lambda _: slice(0, 1))],
        remainder=FunctionTransformer(),
        sparse_threshold=0.0,
    )

    with pytest.raises(TypeError, match="other than a list of keys"):
        transformer.fit_transform(X)


def test__order_preserving_column_transformer__permuted_full_selection() -> None:
    """Every column selected, in another order, is still restored to the input's."""
    X = pd.DataFrame({"a": ["x", "y", "z"], "b": ["q", "p", "r"], "c": ["v", "v", "u"]})
    transformer = OrderPreservingColumnTransformer(
        # The whole input, in an order of its own: `ColumnTransformer` returns one block
        # holding every column, so the gather that undoes it covers all of them.
        transformers=[("encoder", OrdinalEncoder(), ["c", "a", "b"])],
        remainder=FunctionTransformer(),
        sparse_threshold=0.0,
    )
    encoded = [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [2.0, 2.0, 0.0]]

    np.testing.assert_array_equal(transformer.fit_transform(X), encoded)
    # `transform` leaves the values to sklearn, so it reaches the reorder by its own
    # route -- the assembly above never builds the shuffled order to begin with.
    np.testing.assert_array_equal(transformer.transform(X), encoded)


def test__order_preserving_column_transformer__already_in_input_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stack that is already in input order is handed back without a gather."""
    stacked = []
    hstack = ColumnTransformer._hstack

    def recording(self: ColumnTransformer, Xs: list, **kwargs: object) -> np.ndarray:
        stacked.append(hstack(self, Xs, **kwargs))
        return stacked[-1]

    monkeypatch.setattr(ColumnTransformer, "_hstack", recording)

    # The encoded column leads, so `ColumnTransformer`'s own order is the input's. The
    # int64 column is what keeps this off the assembly path, which reaches its result
    # without a stack to gather over at all.
    X = pd.DataFrame({"a": ["x", "y"], "b": [1, 2], "c": [3.0, 4.0]})
    transformer = OrderPreservingColumnTransformer(
        transformers=[("encoder", OrdinalEncoder(), ["a"])],
        remainder=FunctionTransformer(),
        sparse_threshold=0.0,
    )

    out = transformer.fit_transform(X)

    np.testing.assert_array_equal(out, [[0.0, 1.0, 3.0], [1.0, 2.0, 4.0]])
    assert out is stacked[-1]


def test__order_preserving_column_transformer__skipped_gather_keeps_layout() -> None:
    """Skipping a gather that changes no position must not change the layout either.

    `_assembled_order` treats the contiguity of what leaves here as load-bearing -- the
    SVD downstream settles on a different basis per layout -- and the gather leaves a
    Fortran-contiguous array whatever it was handed. So must skipping it.
    """
    rows = 64
    # The encoded columns lead, so `ColumnTransformer`'s own order is already the
    # input's and there is nothing for the gather to move. The object column is what
    # keeps this off the assembly path *and* what makes the stack row-major, so the two
    # layouts differ here where they would otherwise coincide.
    X = pd.DataFrame(
        {
            "a": pd.Series(["x", "y"] * (rows // 2), dtype="string"),
            "b": pd.Series(["p", "q"] * (rows // 2), dtype="string"),
            "c": np.arange(rows, dtype=object),
            "d": np.arange(rows, dtype=float),
        }
    )
    transformer = get_ordinal_encoder()

    out = transformer.fit_transform(X)

    assert out.flags.f_contiguous
    np.testing.assert_array_equal(out, transformer.transform(X))


def test__pipeline__num_added_features():
    """Test that the pipeline returns the correct number of added features."""
    pipeline = PreprocessingPipeline(
        steps=[
            ReshapeFeatureDistributionsStep(
                transform_name="quantile_uni",
                append_to_original="auto",
                random_state=42,
                max_features_per_estimator=500,
            ),
            AddFingerprintFeaturesStep(),
        ]
    )
    assert pipeline.num_added_features(100, _get_schema(num_columns=10)) == 11
    assert pipeline.num_added_features(100, _get_schema(num_columns=501)) == 1

    pipeline = PreprocessingPipeline(
        steps=[
            ReshapeFeatureDistributionsStep(
                transform_name="quantile_uni",
                append_to_original="auto",
                random_state=42,
                max_features_per_estimator=500,
            ),
            AddSVDFeaturesStep(global_transformer_name="svd", random_state=42),
        ]
    )
    # Reshape adds 10 (append_to_original), then SVD sees 20 features and adds
    # min(100//10+1, 20//2) = min(11, 10) = 10. Total added = 20.
    assert pipeline.num_added_features(100, _get_schema(num_columns=10)) == 10 + 10

    pipeline = PreprocessingPipeline(
        steps=[
            RemoveConstantFeaturesStep(),
            AddFingerprintFeaturesStep(),
        ]
    )
    # Note that we currently don't count the removed features as -1.
    # This is a minor effect that we ignore for now. In the future,
    # we will make sure that the pipeline never actually sees constant features.
    assert pipeline.num_added_features(100, _get_schema(num_columns=10)) == 1


# ---------------------------------------------------------------------------
# EfficientColumnTransformer against the ColumnTransformer it stands in for
# ---------------------------------------------------------------------------
#
# The class exists only to reach `ColumnTransformer`'s result with fewer full-size
# arrays, so sklearn is the oracle throughout: every case below compares against a
# plain `ColumnTransformer` over the same transformers, and says whether the assembly
# was taken or fallen back from -- a fallback that stops being a fallback is a silent
# behaviour change, and an assembly that quietly stops being one is a silent
# regression.


def _reference(
    transformer: EfficientColumnTransformer, X: XType, y: YType = None
) -> np.ndarray:
    """What a stock `ColumnTransformer` over the same transformers produces."""
    plain = ColumnTransformer(
        list(transformer.transformers),
        remainder=transformer.remainder,
        sparse_threshold=transformer.sparse_threshold,
    )
    return plain.fit_transform(X, y)


@pytest.fixture
def assemblies(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, ...]]:
    """Records the shape of every array the assembly writes into."""
    written: list[tuple[int, ...]] = []
    assemble = EfficientColumnTransformer._assemble

    def recording(self: EfficientColumnTransformer, *args: object) -> np.ndarray:
        out = assemble(self, *args)
        written.append(out.shape)
        return out

    monkeypatch.setattr(EfficientColumnTransformer, "_assemble", recording)
    return written


def _mixed_frame() -> pd.DataFrame:
    """Numeric and string columns, alternating, all passthrough columns float64."""
    return pd.DataFrame(
        {
            "n0": np.arange(6, dtype="float64"),
            "s0": pd.Series(list("abcabc"), dtype="string"),
            "n1": np.arange(6, dtype="float64") * 2,
            "s1": pd.Series(list("xyzxyz"), dtype="string"),
        }
    )


def test__efficient_column_transformer__assembles_the_column_transformer_layout(
    assemblies: list[tuple[int, ...]],
) -> None:
    """Codes first, then the columns the encoder left, in input order."""
    X = np.column_stack([np.arange(6.0), np.tile([1.0, 2.0], 3), np.arange(6.0) * 3])
    transformer = EfficientColumnTransformer(
        [("encoder", OrdinalEncoder(), [1])], remainder="passthrough"
    )

    out = transformer.fit_transform(X)

    assert assemblies == [(6, 3)]
    np.testing.assert_array_equal(out, _reference(transformer, X))


def test__efficient_column_transformer__assembles_the_preserved_layout(
    assemblies: list[tuple[int, ...]],
) -> None:
    """Every column at its own position, which is what the reference has to be gathered
    back into.
    """
    X = _mixed_frame()
    transformer = get_ordinal_encoder()

    out = transformer.fit_transform(X)

    assert assemblies == [(6, 4)]
    # The reference is the stacked layout: encoded columns first, then the rest.
    reference = _reference(transformer, X)
    np.testing.assert_array_equal(out[:, [1, 3]], reference[:, [0, 1]])
    np.testing.assert_array_equal(out[:, [0, 2]], reference[:, [2, 3]])


@pytest.mark.parametrize(
    ("transformers", "kwargs", "why"),
    [
        pytest.param(
            [("onehot", OneHotEncoder(sparse_output=False), [1])],
            {},
            "an expanding transformer has no column to write each output into",
            id="expanding-transformer",
        ),
        pytest.param(
            [("a", OrdinalEncoder(), [0]), ("b", OrdinalEncoder(), [1])],
            {},
            "two transformers can each claim a column",
            id="two-transformers",
        ),
        pytest.param(
            [("encoder", OrdinalEncoder(), [1])],
            {"remainder": "drop"},
            "a dropped remainder does not cover the output",
            id="dropping-remainder",
        ),
        pytest.param(
            [("encoder", OrdinalEncoder(), slice(1, 2))],
            {"remainder": "passthrough"},
            "a slice selection is not a list of column keys",
            id="slice-selection",
        ),
    ],
)
def test__efficient_column_transformer__falls_back_and_still_matches_sklearn(
    transformers: list[tuple],
    kwargs: dict,
    why: str,
    assemblies: list[tuple[int, ...]],
) -> None:
    """A shape the assembly cannot place has to come out of sklearn unchanged."""
    X = np.column_stack([np.arange(6.0), np.tile([1.0, 2.0], 3), np.arange(6.0) * 3])
    transformer = EfficientColumnTransformer(transformers, **kwargs)

    out = transformer.fit_transform(X)

    assert assemblies == [], why
    np.testing.assert_array_equal(out, _reference(transformer, X))


def test__efficient_column_transformer__a_y_falls_back(
    assemblies: list[tuple[int, ...]],
) -> None:
    """The one-row fit has nothing to line a full-length `y` up with, so decline it."""
    X = np.column_stack([np.arange(6.0), np.tile([1.0, 2.0], 3)])
    y = np.arange(6.0)
    transformer = EfficientColumnTransformer(
        [("encoder", OrdinalEncoder(), [1])], remainder="passthrough"
    )

    out = transformer.fit_transform(X, y)

    assert assemblies == []
    np.testing.assert_array_equal(out, _reference(transformer, X, y))


def test__order_preserving_column_transformer__declines_an_object_passthrough(
    assemblies: list[tuple[int, ...]],
) -> None:
    """A frame the assembly cannot write column by column still goes through sklearn.

    `object` is not in the encoder's dtype selection, so it reaches the output through
    the passthrough; writing it into a float64 array is not the conversion sklearn's own
    path would have done.
    """
    frame = _mixed_frame().assign(o=np.arange(6).astype(object))
    transformer = get_ordinal_encoder()

    out = transformer.fit_transform(frame)

    assert assemblies == []
    assert out.shape == frame.shape


def test__efficient_column_transformer__declines_duplicate_column_labels() -> None:
    """Each column is placed by key, so a duplicated one makes its place ambiguous.

    Asserted on the check rather than end to end: sklearn's own column selector refuses
    such a frame first, and this is what keeps the assembly from being the thing that
    has to.
    """
    frame = _mixed_frame()
    transformer = get_ordinal_encoder()
    transformer.fit(frame)

    doubled = frame.iloc[:, [0, 1, 2, 3, 0]]
    assert not transformer._can_assemble(
        doubled, doubled.iloc[:1], transformer.selected_columns()
    )


def test__efficient_column_transformer__fit_builds_no_full_size_output() -> None:
    """`fit` learns everything `ColumnTransformer.fit` does without transforming.

    `ColumnTransformer.fit` is implemented as `fit_transform`, so it pays for a
    full-size result and throws it away. This asserts the state is identical anyway --
    it is kept on the estimator and drives `transform` -- and that nothing wider than
    the one bookkeeping row was ever stacked.
    """
    X = np.column_stack([np.arange(6.0), np.tile([1.0, 2.0], 3), np.arange(6.0) * 3])
    transformer = EfficientColumnTransformer(
        [("encoder", OrdinalEncoder(), [1])], remainder="passthrough"
    )
    stacked: list[int] = []
    hstack = ColumnTransformer._hstack

    def recording(self: ColumnTransformer, Xs: list, **kwargs: object) -> np.ndarray:
        stacked.append(max(len(x) for x in Xs))
        return hstack(self, Xs, **kwargs)

    with mock.patch.object(ColumnTransformer, "_hstack", recording):
        transformer.fit(X)

    assert stacked == [1], "a full-size result was stacked and discarded"

    reference = ColumnTransformer(
        list(transformer.transformers), remainder=transformer.remainder
    ).fit(X)
    assert transformer.n_features_in_ == reference.n_features_in_
    assert transformer.output_indices_ == reference.output_indices_
    for learned, wanted in zip(
        transformer.named_transformers_["encoder"].categories_,
        reference.named_transformers_["encoder"].categories_,
        strict=True,
    ):
        np.testing.assert_array_equal(learned, wanted)


def test__efficient_column_transformer__transform_keeps_sklearns_validation() -> None:
    """`transform` is sklearn's, so what it refuses at predict time stays refused."""
    X = np.column_stack([np.arange(6.0), np.tile([1.0, 2.0], 3), np.arange(6.0) * 3])
    transformer = EfficientColumnTransformer(
        [("encoder", OrdinalEncoder(), [1])], remainder="passthrough"
    )
    transformer.fit_transform(X)

    np.testing.assert_array_equal(transformer.transform(X), _reference(transformer, X))
    with pytest.raises(ValueError, match="features"):
        transformer.transform(X[:, :2])


def test__efficient_column_transformer__clones_like_a_column_transformer() -> None:
    """No parameter may be hidden from `get_params`, or sklearn's `clone` drops it.

    Which is the reason the class declares no `__init__` of its own: sklearn reads the
    parameter names off that signature, and one taking `**kwargs` reports only what it
    names -- everything else is silently lost on a clone.
    """
    transformers = [("encoder", OrdinalEncoder(), [1])]
    transformer = EfficientColumnTransformer(
        transformers,
        remainder="passthrough",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )

    clone = sklearn.base.clone(transformer)

    assert (
        transformer.get_params(deep=False).keys()
        == ColumnTransformer(transformers).get_params(deep=False).keys()
    )
    assert clone.remainder == "passthrough"
    assert clone.sparse_threshold == 0.0
    assert clone.verbose_feature_names_out is False
    assert clone.preserves_column_order is transformer.preserves_column_order


# --- ownership of the assembled output --------------------------------------------
#
# The assembly hands its array straight to the caller, who writes into it, so it has to
# own it. How many copies that takes depends on the frame's block layout, which is what
# these pin. The frames here are all float64, so the encoder selects nothing and the
# whole step is the cast -- the case where taking `to_numpy`'s result would have been
# tempting.


def _fragmented_float_frame(values: np.ndarray) -> pd.DataFrame:
    """A float frame held as one block per column, as a recast frame is left."""
    frame = pd.DataFrame(values)
    frame[frame.columns] = frame[frame.columns].astype(values.dtype)
    return frame


@pytest.mark.parametrize("fit", [True, False], ids=["fit_transform", "transform"])
def test__order_preserving_column_transformer__owns_a_multi_block_frames_output(
    *, fit: bool
) -> None:
    """A frame of many blocks is assembled directly, never routed through `to_numpy`.

    Asserted through the frame rather than the result, because what the result looks
    like depends on the pandas: `to_numpy` consolidates the frame in place before
    pandas 3, so a run that went through it would cost a second full-size buffer and
    leave the frame holding one block instead of many.
    """
    values = np.random.default_rng(0).standard_normal((8, 5))
    frame = _fragmented_float_frame(values)
    transformer = get_ordinal_encoder()

    out = (
        transformer.fit_transform(frame)
        if fit
        else transformer.fit(frame).transform(frame)
    )

    np.testing.assert_array_equal(out, values)
    assert out.flags.writeable
    assert out.flags.f_contiguous
    assert not np.shares_memory(out, values)
    # Still one block per column: nothing consolidated it on the way.
    assert len(frame._mgr.blocks) == frame.shape[1]
    # And the buffer is the caller's alone -- writing into it cannot reach the frame.
    assert not any(
        np.shares_memory(out, frame.iloc[:, position].to_numpy(copy=False))
        for position in range(frame.shape[1])
    )


@pytest.mark.parametrize("fit", [True, False], ids=["fit_transform", "transform"])
def test__order_preserving_column_transformer__owns_a_single_block_frames_output(
    *, fit: bool
) -> None:
    """A single-block frame is handed out by pandas as a view of its own buffer."""
    values = np.random.default_rng(0).standard_normal((8, 5))
    frame = pd.DataFrame(values, copy=False)
    assert len(frame._mgr.blocks) == 1
    assert np.shares_memory(frame.to_numpy(dtype=np.float64, copy=False), values)
    transformer = get_ordinal_encoder()

    out = (
        transformer.fit_transform(frame)
        if fit
        else transformer.fit(frame).transform(frame)
    )

    np.testing.assert_array_equal(out, values)
    assert out.flags.writeable
    assert out.flags.f_contiguous
    # Writing into the result must not reach the caller's array. Under copy-on-write
    # the view pandas hands out is read-only, but on pandas 2 it is writeable and
    # aliases `values`.
    assert not np.shares_memory(out, values)
    out[0, 0] = 12345.0
    assert values[0, 0] != 12345.0


def test__efficient_column_transformer__an_unfitted_transform_is_not_an_identity() -> (
    None
):
    """A transformer that never selected anything because it was never fitted.

    It has no selection to read, which must not be mistaken for having selected no
    columns -- that would hand the input back untransformed instead of failing.
    """
    with pytest.raises(NotFittedError):
        get_ordinal_encoder().transform(_mixed_frame())


def test__efficient_column_transformer__a_validating_remainder_is_not_a_passthrough(
    assemblies: list[tuple[int, ...]],
) -> None:
    """`validate=True` gives the remainder real work, so it cannot be assembled around.

    It sends the columns it is handed through `check_array`, which coerces their dtype
    and refuses a NaN. A table sklearn would reject therefore has to keep being
    rejected, rather than quietly assembled past the check.
    """
    X = np.column_stack([np.arange(6.0), np.tile([1.0, 2.0], 3), np.arange(6.0) * 3])
    X[0, 2] = np.nan
    transformer = EfficientColumnTransformer(
        [("encoder", OrdinalEncoder(), [1])],
        remainder=FunctionTransformer(validate=True),
    )

    with pytest.raises(ValueError, match="NaN"):
        transformer.fit_transform(X)
    assert assemblies == []


# --- what the assembly declines -------------------------------------------------
#
# Three shapes reach the same result through sklearn only. What each has in common is
# that the assembly would not be wrong so much as silently different -- a dense array
# where a sparse one was asked for, a name where a value was, a routed parameter
# nobody looked at.


@pytest.mark.parametrize("globally", [False, True], ids=["set_output", "config"])
def test__efficient_column_transformer__declines_a_frame_output(
    assemblies: list[tuple[int, ...]], *, globally: bool
) -> None:
    """A caller asking for a frame gets one, from sklearn, not an array from here.

    Either route to it: the estimator's own `set_output`, or sklearn's global config,
    which a surrounding `config_context` turns on without touching this estimator at
    all.
    """
    X = _mixed_frame()
    transformer = get_ordinal_encoder()
    if not globally:
        transformer.set_output(transform="pandas")

    with sklearn.config_context(transform_output="pandas" if globally else "default"):
        out = transformer.fit_transform(X)

    assert assemblies == []
    assert isinstance(out, pd.DataFrame)
    # in the input's order, values and names alike
    assert list(out.columns) == list(X.columns)
    np.testing.assert_array_equal(out["n0"], X["n0"])
    np.testing.assert_array_equal(out["n1"], X["n1"])


def test__efficient_column_transformer__names_the_stacked_order_it_returns(
    assemblies: list[tuple[int, ...]],
) -> None:
    """The plain class returns sklearn's own order, so it keeps sklearn's own names.

    Nothing is overridden here, and nothing may be: the reorder in the subclass is
    what makes the parent's names wrong, and an override hoisted up to this class
    would name every column after whatever that reorder would have moved there. What
    keeps the two in step is that the assembly places the transformed columns by
    `output_indices_`, which is the same state the names are read off.
    """
    X = _mixed_frame()
    transformers = [("encoder", OrdinalEncoder(), ["s0", "s1"])]
    kwargs = {
        "remainder": "passthrough",
        "sparse_threshold": 0.0,
        "verbose_feature_names_out": False,
    }
    transformer = EfficientColumnTransformer(transformers, **kwargs)

    out = transformer.fit_transform(X)
    names = list(transformer.get_feature_names_out())

    assert assemblies == [(6, 4)]
    assert names == list(
        ColumnTransformer(transformers, **kwargs).fit(X).get_feature_names_out()
    )
    # And they describe the columns they are on, which is the whole point of them.
    named = pd.DataFrame(out, columns=names)
    for column in ["n0", "n1"]:
        np.testing.assert_array_equal(named[column], X[column])


def test__order_preserving_column_transformer__names_the_order_it_returns() -> None:
    """The names have to be reordered with the columns, not left in sklearn's order.

    Nothing here reads them, but sklearn's `set_output` wrapper does: it labels a
    frame output with exactly these, over data this class has already reordered. Left
    alone, every column would be named after whatever used to sit at its position.
    """
    X = _mixed_frame()
    # A named remainder, since `FunctionTransformer` cannot name its own columns and
    # sklearn refuses to name the output at all through one.
    transformer = OrderPreservingColumnTransformer(
        [("encoder", OrdinalEncoder(), ["s0", "s1"])],
        remainder="passthrough",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    ).fit(X)

    assert list(transformer.get_feature_names_out()) == list(X.columns)
    # The stacked order is still what the parent reports, and what has to be undone.
    assert list(ColumnTransformer.get_feature_names_out(transformer)) == [
        "s0",
        "s1",
        "n0",
        "n1",
    ]


def test__efficient_column_transformer__declines_routed_metadata(
    assemblies: list[tuple[int, ...]],
) -> None:
    """A parameter meant for the inner transformer must not be quietly dropped."""
    X = _mixed_frame()
    transformer = get_ordinal_encoder()

    # Left to sklearn to accept or refuse -- which without metadata routing enabled is
    # to refuse: a `ValueError` saying so, or, on the sklearn 1.2 floor whose
    # `fit_transform` takes no such parameter at all, a `TypeError` from the call.
    # What must not happen is an assembly that ignores it.
    with pytest.raises((TypeError, ValueError), match="sample_weight"):
        transformer.fit_transform(X, None, encoder__sample_weight=np.ones(6))

    assert assemblies == []


def test__efficient_column_transformer__declines_a_sparse_input(
    assemblies: list[tuple[int, ...]],
) -> None:
    """A sparse input keeps sklearn's sparse output, which the assembly cannot write.

    The same transformer over the same values densely *is* assembled, so sparseness is
    the only reason for the fallback here.
    """
    values = np.zeros((10, 4))
    values[0, 0], values[3, 1], values[5, 2] = 1.0, 2.0, 3.0
    transformers = [("scaler", MaxAbsScaler(), [1])]

    dense = EfficientColumnTransformer(transformers, remainder="passthrough")
    dense.fit_transform(values)
    assert assemblies == [(10, 4)], "the dense input is one the assembly takes"

    transformer = EfficientColumnTransformer(transformers, remainder="passthrough")
    out = transformer.fit_transform(sparse.csr_matrix(values))

    assert assemblies == [(10, 4)], "a sparse input was assembled"
    assert transformer.sparse_output_
    assert sparse.issparse(out)
    np.testing.assert_array_equal(
        out.toarray(), _reference(transformer, sparse.csr_matrix(values)).toarray()
    )


def test__efficient_column_transformer__fit_falls_back_to_sklearns_state(
    assemblies: list[tuple[int, ...]],
) -> None:
    """A `fit` the assembly declines has to leave exactly the state sklearn leaves.

    The one-row probe fit runs first and is then thrown away, so what the estimator
    carries into `transform` is the full fit's -- categories learned from every row
    included.
    """
    frame = _mixed_frame().assign(o=np.arange(6).astype(object))
    transformer = get_ordinal_encoder().fit(frame)
    reference = ColumnTransformer(
        list(transformer.transformers),
        remainder=transformer.remainder,
        sparse_threshold=transformer.sparse_threshold,
    ).fit(frame)

    assert assemblies == [], "an object passthrough column was assembled"
    assert transformer.n_features_in_ == reference.n_features_in_
    assert transformer.output_indices_ == reference.output_indices_
    for learned, wanted in zip(
        transformer.named_transformers_["encoder"].categories_,
        reference.named_transformers_["encoder"].categories_,
        strict=True,
    ):
        np.testing.assert_array_equal(learned, wanted)


# --- the assembly and the fallback are the same output ----------------------------
#
# `fit_transform` assembles where `transform` goes through sklearn, so the two routes
# have to arrive at the same array -- values, dtype and memory layout. Fit and predict
# run one each, and a layout that differs between them rotates the SVD's basis
# downstream on exactly one of the two.


@pytest.mark.parametrize("selected", [[1], [0], [0, 2], [1, 2], [0, 1, 2]], ids=str)
def test__efficient_column_transformer__transform_repeats_the_assembly(
    selected: list[int], assemblies: list[tuple[int, ...]]
) -> None:
    """Every selection over an array, assembled at fit and stacked at transform."""
    X = np.column_stack([np.arange(6.0), np.tile([1.0, 2.0], 3), np.arange(6.0) * 3])
    transformer = EfficientColumnTransformer(
        [("encoder", OrdinalEncoder(), selected)], remainder="passthrough"
    )

    assembled = transformer.fit_transform(X)
    stacked = transformer.transform(X)

    assert assemblies == [(6, 3)], "the fit was not assembled"
    np.testing.assert_array_equal(assembled, stacked)
    assert assembled.dtype == stacked.dtype
    assert np.isfortran(assembled) == np.isfortran(stacked)


def test__order_preserving_column_transformer__transform_repeats_the_assembly(
    assemblies: list[tuple[int, ...]],
) -> None:
    """The same, where the fallback reaches the input's order through a gather."""
    X = _mixed_frame()
    transformer = get_ordinal_encoder()

    assembled = transformer.fit_transform(X)
    gathered = transformer.transform(X)

    assert assemblies == [(6, 4)], "the fit was not assembled"
    np.testing.assert_array_equal(assembled, gathered)
    assert assembled.dtype == gathered.dtype
    assert np.isfortran(assembled) == np.isfortran(gathered)


@pytest.mark.parametrize("selected", [["s0"], ["s0", "s1"]], ids=str)
def test__efficient_column_transformer__assembles_a_frame_in_the_stacked_layout(
    selected: list[str], assemblies: list[tuple[int, ...]]
) -> None:
    """A frame through the plain class, whose output order is sklearn's own.

    Every other frame case here goes through the order-preserving subclass, so this is
    what pins the other half of `_output_positions` -- the transformed columns in their
    own slice -- for an input whose columns are keyed by name.

    Both widths of selection, because they land on different layouts: sklearn stacks a
    column-major output only when the encoder's own block is column-major too, which
    one column is and two are not.
    """
    # Only the selected string columns: an unselected one is not float64, which the
    # assembly declines outright.
    X = _mixed_frame().drop(columns=[s for s in ("s0", "s1") if s not in selected])
    transformer = EfficientColumnTransformer(
        [("encoder", OrdinalEncoder(), selected)],
        remainder=FunctionTransformer(),
        sparse_threshold=0.0,
    )

    out = transformer.fit_transform(X)
    reference = _reference(transformer, X)

    assert assemblies == [(6, X.shape[1])]
    np.testing.assert_array_equal(out, reference)
    assert out.dtype == reference.dtype
    assert np.isfortran(out) == np.isfortran(reference)


def test__order_preserving_column_transformer__promotes_a_narrower_encoder(
    assemblies: list[tuple[int, ...]],
) -> None:
    """A float32 encoder against float64 passthrough columns: sklearn promotes."""
    X = _mixed_frame()
    transformer = get_ordinal_encoder(numpy_dtype=np.float32)  # type: ignore[arg-type]

    out = transformer.fit_transform(X)

    assert assemblies == [(6, 4)]
    assert out.dtype == _reference(transformer, X).dtype == np.float64


# --- the shared identity predicate ------------------------------------------------
#
# Two steps decide from it that a transform need not run: the assembly skips a
# remainder it vouches for, and the reshape step drops its ColumnTransformer for a
# gather. So it has to answer for the cases where that is provable and decline the rest.


class _SneakyFunctionTransformer(FunctionTransformer):
    """A subclass that transforms without a `func`, which no attribute would reveal."""

    @override
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X * 2


@pytest.mark.parametrize(
    ("transformer", "identity", "why"),
    [
        pytest.param("passthrough", True, "says so outright", id="passthrough"),
        pytest.param("drop", False, "drops its columns", id="drop"),
        pytest.param(FunctionTransformer(), True, "no func, no validation", id="bare"),
        pytest.param(
            FunctionTransformer(validate=True),
            False,
            "check_array coerces the dtype and refuses a NaN",
            id="validating",
        ),
        pytest.param(
            FunctionTransformer(func=np.sqrt), False, "has a func", id="with-func"
        ),
        pytest.param(
            FunctionTransformer(func=None, inverse_func=np.sqrt),
            True,
            "an inverse is not on the forward path",
            id="with-inverse-only",
        ),
        pytest.param(
            _SneakyFunctionTransformer(),
            False,
            "a subclass may override transform with func still unset",
            id="subclass",
        ),
        pytest.param(OrdinalEncoder(), False, "encodes its columns", id="encoder"),
    ],
)
def test__is_identity_transformer(
    transformer: object, identity: bool, why: str
) -> None:
    """Only what provably hands its input back counts as the identity."""
    assert is_identity_transformer(transformer) is identity, why


def test__is_identity_transformer__is_what_both_callers_use() -> None:
    """The two steps that skip work on it must not drift back to their own copies."""
    encoder = get_ordinal_encoder()
    assert is_identity_transformer(encoder.remainder)

    with mock.patch(
        "tabpfn.preprocessing.steps.preprocessing_helpers.is_identity_transformer",
        return_value=False,
    ):
        # Declining the remainder has to cost the assembly, not go unnoticed.
        assert not encoder._may_assemble(_mixed_frame(), {})
