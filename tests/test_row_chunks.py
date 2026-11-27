#  Copyright (c) Prior Labs GmbH 2026.

"""A fit above the row limit splits the training rows into chunks."""

from __future__ import annotations

import pathlib
import types

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

import tabpfn.base
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.base import (
    drop_row_chunks,
    member_subsample_samples,
    plan_row_chunks,
    repeat_for_row_chunks,
    validate_additional_model_input,
)
from tabpfn.errors import TabPFNValidationError
from tabpfn.model_loading import load_fitted_tabpfn_model, save_fitted_tabpfn_model
from tabpfn.preprocessing.configs import (
    FeatureSubsamplingMethod,
    PreprocessorConfig,
    RegressorEnsembleConfig,
)
from tabpfn.preprocessing.datamodel import FeatureSchema
from tabpfn.preprocessing.ensemble import (
    TabPFNEnsemblePreprocessor,
    generate_classification_ensemble_configs,
    generate_regression_ensemble_configs,
)

LIMIT = 50
N_ROWS = 120


def _estimator(
    estimator_type: str = "regressor",
    *,
    subsample_samples: int | None = None,
    ignore_pretraining_limits: bool = False,
) -> types.SimpleNamespace:
    """What `plan_row_chunks` reads of an estimator."""
    return types.SimpleNamespace(
        inference_config_=types.SimpleNamespace(
            MAX_NUMBER_OF_SAMPLES=LIMIT, SUBSAMPLE_SAMPLES=subsample_samples
        ),
        ignore_pretraining_limits=ignore_pretraining_limits,
        estimator_type=estimator_type,
    )


def _data(estimator_type: str, n_rows: int = N_ROWS) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_rows, 4))
    if estimator_type == "classifier":
        return X, (X[:, 0] > 0).astype(int) + (X[:, 1] > 1).astype(int)
    return X, X @ rng.normal(size=4)


def _split_regressor(**kwargs) -> TabPFNRegressor:
    return TabPFNRegressor(
        n_estimators=2,
        device="cpu",
        random_state=0,
        inference_config={"MAX_NUMBER_OF_SAMPLES": LIMIT},
        **kwargs,
    )


def test__plan_row_chunks__within_the_limit__no_chunks() -> None:
    X = np.zeros((LIMIT, 1))
    assert plan_row_chunks(_estimator(), X, X[:, 0], random_state=0) is None


@pytest.mark.parametrize("estimator_type", ["regressor", "classifier"])
def test__plan_row_chunks__no_rows__left_to_the_validation(estimator_type: str) -> None:
    X = np.zeros((0, 1))
    assert (
        plan_row_chunks(_estimator(estimator_type), X, X[:, 0], random_state=0) is None
    )


def test__plan_row_chunks__above_the_limit__fewest_disjoint_chunks_within_it() -> None:
    with pytest.warns(UserWarning, match="split into 3 chunks"):
        chunks = plan_row_chunks(
            _estimator(), np.zeros((N_ROWS, 1)), np.zeros(N_ROWS), random_state=0
        )
    assert [len(chunk) for chunk in chunks] == [40, 40, 40]
    assert np.array_equal(np.sort(np.concatenate(chunks)), np.arange(N_ROWS))


def test__plan_row_chunks__classifier__class_counts_differ_by_at_most_one() -> None:
    y = np.repeat(["a", "b", "c"], [70, 41, 9])
    with pytest.warns(UserWarning, match="split into 3 chunks"):
        chunks = plan_row_chunks(
            _estimator("classifier"), np.zeros((len(y), 1)), y, random_state=0
        )
    assert np.array_equal(np.sort(np.concatenate(chunks)), np.arange(len(y)))
    for label, count in (("a", 70), ("b", 41), ("c", 9)):
        counts = [int(np.sum(y[chunk] == label)) for chunk in chunks]
        assert sum(counts) == count
        assert max(counts) - min(counts) <= 1


def test__plan_row_chunks__class_with_fewer_rows_than_chunks__in_every_chunk() -> None:
    y = np.repeat(["a", "b", "c"], [100, 18, 2])
    with pytest.warns(UserWarning, match="split into 3 chunks"):
        chunks = plan_row_chunks(
            _estimator("classifier"), np.zeros((len(y), 1)), y, random_state=0
        )
    for chunk in chunks:
        assert set(y[chunk]) == {"a", "b", "c"}
        # One row of "c", as the stratified row sampler keeps one of every class.
        assert int(np.sum(y[chunk] == "c")) == 1
        assert len(chunk) <= LIMIT
    assert np.array_equal(np.unique(np.concatenate(chunks)), np.arange(len(y)))


def test__plan_row_chunks__rare_classes_before_a_large_one__one_row_each() -> None:
    y = np.repeat(["r1", "r2", "big"], [2, 1, 117])
    with pytest.warns(UserWarning, match="split into 3 chunks"):
        chunks = plan_row_chunks(
            _estimator("classifier"), np.zeros((len(y), 1)), y, random_state=0
        )
    for chunk in chunks:
        assert int(np.sum(y[chunk] == "r1")) == int(np.sum(y[chunk] == "r2")) == 1
    assert np.array_equal(np.unique(np.concatenate(chunks)), np.arange(len(y)))


def test__plan_row_chunks__rows_in_every_chunk_need_room__one_more_chunk() -> None:
    # 150 rows fill 3 chunks of 50, so the 2 rows of "b" for every chunk need a 4th.
    y = np.repeat(["a", "b"], [148, 2])
    with pytest.warns(UserWarning, match="split into 4 chunks"):
        chunks = plan_row_chunks(
            _estimator("classifier"), np.zeros((len(y), 1)), y, random_state=0
        )
    assert all(len(chunk) <= LIMIT for chunk in chunks)
    assert all(set(y[chunk]) == {"a", "b"} for chunk in chunks)


@pytest.mark.parametrize(
    "estimator",
    [_estimator(subsample_samples=30), _estimator(ignore_pretraining_limits=True)],
    ids=["subsample_samples", "ignore_pretraining_limits"],
)
def test__plan_row_chunks__rows_set_by_the_caller__no_chunks(
    estimator: types.SimpleNamespace,
) -> None:
    X = np.zeros((N_ROWS, 1))
    assert plan_row_chunks(estimator, X, X[:, 0], random_state=0) is None


def test__plan_row_chunks__labels_too_sparse_to_share__left_to_the_validation() -> None:
    # A continuous target given to a classifier: every class has a single row.
    X, y = np.zeros((N_ROWS, 1)), np.arange(N_ROWS)
    assert plan_row_chunks(_estimator("classifier"), X, y, random_state=0) is None


def test__plan_row_chunks__unhashable_labels__left_to_the_validation() -> None:
    X = np.zeros((N_ROWS, 1))
    y = np.empty(N_ROWS, dtype=object)
    y[:] = [[0, 1]] * N_ROWS
    assert plan_row_chunks(_estimator("classifier"), X, y, random_state=0) is None


def test__plan_row_chunks__labels_of_two_columns__left_to_the_validation() -> None:
    X, y = np.zeros((N_ROWS, 1)), np.zeros((N_ROWS, 2))
    assert plan_row_chunks(_estimator("classifier"), X, y, random_state=0) is None


def test__plan_row_chunks__no_limit_but_ignore_pretraining_limits__no_chunks() -> None:
    estimator = _estimator(ignore_pretraining_limits=True)
    estimator.inference_config_.MAX_NUMBER_OF_SAMPLES = 0
    X = np.zeros((N_ROWS, 1))
    assert plan_row_chunks(estimator, X, X[:, 0], random_state=0) is None


def test__plan_row_chunks__y_of_another_length__left_to_the_validation() -> None:
    X, y = np.zeros((N_ROWS, 1)), np.zeros(N_ROWS - 1)
    assert plan_row_chunks(_estimator("classifier"), X, y, random_state=0) is None


def test__repeat_for_row_chunks__every_chunk_gets_the_ensemble_of_an_unsplit_fit() -> (
    None
):
    configs = generate_regression_ensemble_configs(
        num_estimators=4,
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_decoder="shuffle",
        preprocessor_configs=[
            PreprocessorConfig("none"),
            PreprocessorConfig("safepower"),
        ],
        target_transforms=[None, StandardScaler()],
        random_state=0,
        num_models=2,
        outlier_removal_std=None,
        target_sign_flip=True,
    )
    chunks = [np.array([0]), np.array([1]), np.array([2])]
    repeated = repeat_for_row_chunks(configs, len(chunks))
    rows = member_subsample_samples(
        types.SimpleNamespace(row_chunks_=chunks), len(repeated)
    )

    def setting(config: RegressorEnsembleConfig) -> tuple:
        """What sets an estimator apart, other than its rows."""
        return (
            config.preprocess_config.name,
            type(config.target_transform).__name__,
            config._model_index,
            config.flip_target,
            config.feature_shift_count,
        )

    assert [config.estimator_id for config in repeated] == list(range(12))
    for chunk in chunks:
        read_by = [c for c, r in zip(repeated, rows, strict=True) if r is chunk]
        assert [setting(c) for c in read_by] == [setting(c) for c in configs]
    # A regressor member fits its target transform in place, so none is shared.
    transforms = [c.target_transform for c in repeated if c.target_transform]
    assert len({id(transform) for transform in transforms}) == len(transforms)


def test__ensemble_preprocessor__split_fit__every_chunk_covers_every_feature() -> None:
    n_features, per_chunk, n_chunks = 50, 3, 3
    configs = generate_classification_ensemble_configs(
        num_estimators=per_chunk,
        add_fingerprint_feature=False,
        polynomial_features="no",
        feature_shift_decoder=None,
        preprocessor_configs=[
            PreprocessorConfig(
                "none", categorical_name="numeric", max_features_per_estimator=20
            )
        ],
        class_shift_method=None,
        n_classes=2,
        random_state=0,
        num_models=1,
        outlier_removal_std=None,
    )
    preprocessor = TabPFNEnsemblePreprocessor(
        configs=repeat_for_row_chunks(configs, n_chunks),
        n_samples=N_ROWS,
        feature_schema=FeatureSchema.from_only_categorical_indices([], n_features),
        random_state=0,
        n_preprocessing_jobs=1,
        feature_subsampling_method=FeatureSubsamplingMethod.BALANCED,
        row_chunks=n_chunks,
    )
    indices = preprocessor.subsample_feature_indices
    for chunk in range(n_chunks):
        block = indices[chunk * per_chunk : (chunk + 1) * per_chunk]
        assert set(np.concatenate(block).tolist()) == set(range(n_features))


@pytest.mark.parametrize("estimator_type", ["regressor", "classifier"])
@pytest.mark.parametrize("by_ranges", [False, True], ids=["whole", "by_ranges"])
def test__fit__above_the_limit__each_chunk_is_read_by_its_own_estimators(
    estimator_type: str, by_ranges: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if by_ranges:
        # One float32 column per range, so the float64 input is cleaned by ranges.
        monkeypatch.setattr(tabpfn.base, "_RANGE_BYTES", N_ROWS * 4)
    X, y = _data(estimator_type)
    model_class = (
        TabPFNClassifier if estimator_type == "classifier" else TabPFNRegressor
    )
    model = model_class(
        n_estimators=2,
        device="cpu",
        random_state=0,
        inference_config={"MAX_NUMBER_OF_SAMPLES": LIMIT},
    )
    with pytest.warns(UserWarning, match="split into 3 chunks") as record:
        model.fit(X, y)
    # The warning points at the caller's `fit` line.
    chunk_warning = next(w for w in record if "chunks" in str(w.message))
    assert chunk_warning.filename == __file__

    rows = model.ensemble_preprocessor_.subsample_row_indices
    assert model.n_estimators_ == len(rows) == 2 * len(model.row_chunks_) == 6
    for member, member_rows in enumerate(rows):
        # The estimators of a chunk share one array, which `row_chunks_` holds.
        assert member_rows is model.row_chunks_[member // 2]
        assert len(member_rows) <= LIMIT
    assert np.array_equal(np.unique(np.concatenate(rows)), np.arange(N_ROWS))
    if estimator_type == "classifier":
        assert np.allclose(model.predict_proba(X[:5]).sum(axis=1), 1)
    else:
        assert np.isfinite(model.predict(X[:5])).all()


def test__fit__ignore_pretraining_limits__every_estimator_reads_all_rows() -> None:
    X, y = _data("regressor")
    model = _split_regressor(ignore_pretraining_limits=True)
    model.fit(X, y)
    assert model.n_estimators_ == 2
    assert model.ensemble_preprocessor_.subsample_row_indices is None


def test__fit__above_the_limit__subsample_samples__raises() -> None:
    X, y = _data("regressor")
    model = TabPFNRegressor(
        n_estimators=2,
        device="cpu",
        inference_config={"MAX_NUMBER_OF_SAMPLES": LIMIT, "SUBSAMPLE_SAMPLES": 30},
    )
    with pytest.raises(TabPFNValidationError, match="Number of samples"):
        model.fit(X, y)


def test__fit__holdout_tuning__every_fold_split_like_the_fit() -> None:
    # Folds of 100 and 101 rows would split into 2 and 3 chunks on their own.
    X, y = _data("classifier", n_rows=126)
    model = TabPFNClassifier(
        n_estimators=2,
        device="cpu",
        random_state=0,
        inference_config={"MAX_NUMBER_OF_SAMPLES": LIMIT},
        tuning_config={
            "calibrate_temperature": True,
            "tuning_holdout_frac": 0.2,
            "tuning_n_folds": 5,
        },
    )
    with pytest.warns(UserWarning, match="split into 3 chunks"):
        model.fit(X, y)
    assert np.allclose(model.predict_proba(X[:5]).sum(axis=1), 1)


def test__get_embeddings__train_rows_of_a_split_fit__raises() -> None:
    X, y = _data("regressor")
    model = _split_regressor()
    with pytest.warns(UserWarning, match="split into 3 chunks"):
        model.fit(X, y)
    with pytest.raises(TabPFNValidationError, match="split its training rows"):
        model.get_embeddings(X[:5], data_source="train")
    assert model.get_embeddings(X[:5]).shape[:2] == (6, 5)


@pytest.mark.parametrize("estimator_type", ["regressor", "classifier"])
def test__save_and_load__split_fit__same_predictions(
    estimator_type: str, tmp_path: pathlib.Path
) -> None:
    X, y = _data(estimator_type)
    model_class = (
        TabPFNClassifier if estimator_type == "classifier" else TabPFNRegressor
    )
    model = model_class(
        n_estimators=2,
        device="cpu",
        random_state=0,
        inference_config={"MAX_NUMBER_OF_SAMPLES": LIMIT},
    )
    with pytest.warns(UserWarning, match="split into 3 chunks"):
        model.fit(X, y)
    path = tmp_path / "model.tabpfn_fit"
    save_fitted_tabpfn_model(model, path)
    loaded = load_fitted_tabpfn_model(path, device="cpu")
    assert loaded.n_estimators_ == 6
    predict = "predict_proba" if estimator_type == "classifier" else "predict"
    np.testing.assert_allclose(
        getattr(loaded, predict)(X[:5]), getattr(model, predict)(X[:5])
    )


def test__validate_additional_model_input__split_fit__raises() -> None:
    def forward(x: object, additional_input: object = None) -> tuple:
        return x, additional_input

    model = types.SimpleNamespace(forward=forward)
    validate_additional_model_input([model], None)
    with pytest.raises(TabPFNValidationError, match="splits the training rows"):
        validate_additional_model_input([model], None, row_chunks=3)


@pytest.mark.parametrize(
    ("n_configs", "kept"), [(8, [0, 1, 2, 3]), (3, [0, 1, 2])], ids=["repeats", "other"]
)
def test__drop_row_chunks__keeps_one_repeat_of_the_ensemble(
    n_configs: int, kept: list[int]
) -> None:
    estimator = types.SimpleNamespace(
        row_chunks_=[np.array([0]), np.array([1])],
        ensemble_configs_=list(range(n_configs)),
    )
    drop_row_chunks(estimator)
    assert estimator.row_chunks_ is None
    assert estimator.ensemble_configs_ == kept


def test__fit__failing_above_the_limit__keeps_the_earlier_fit() -> None:
    X, y = _data("regressor")
    model = _split_regressor()
    model.fit(X[:LIMIT], y[:LIMIT])
    y_with_nan = y.copy()
    y_with_nan[0] = np.nan
    with (
        pytest.warns(UserWarning, match="split into 3 chunks"),
        pytest.raises(TabPFNValidationError, match="NaN"),
    ):
        model.fit(X, y_with_nan)
    assert model.row_chunks_ is None
    assert len(model.ensemble_configs_) == model.n_estimators_ == 2


def test__initialize_dataset_preprocessing__fine_tuning_input__not_split() -> None:
    """Fine-tuning splits its data itself, so it gets no chunks, even after a fit
    that had them.
    """
    X, y = _data("regressor")
    model = _split_regressor()
    with pytest.warns(UserWarning, match="split into 3 chunks"):
        model.fit(X, y)
    with pytest.raises(TabPFNValidationError, match="Number of samples"):
        model._initialize_dataset_preprocessing(X=X, y=y, random_state=0)
    assert model.row_chunks_ is None
