#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for `ImageTransformer`: declaring image columns, embedding them, reapplying."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.exceptions import NotFittedError

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.errors import TabPFNHuggingFaceGatedRepoError, TabPFNValidationError
from tabpfn.inference_config import InferenceConfig
from tabpfn.inference_tuning import ClassifierTuningConfig, RegressorTuningConfig
from tabpfn.preprocessing import images as images_module
from tabpfn.preprocessing.datamodel import INPUT_FEATURE_PREFIX, FeatureModality
from tabpfn.preprocessing.images import DEFAULT_IMAGE_ENCODER_MODEL, ImageTransformer

#: The default width an image column is expanded to.
N_COMPONENTS = InferenceConfig.IMAGE_N_COMPONENTS

#: The width of the default encoder's embedding, which the stub reproduces.
EMBEDDING_DIM = 384


def _stub_encoder(
    payloads: list[bytes], *, model_name: str, device: torch.device
) -> np.ndarray:
    """A deterministic vector per payload: its sha256 digest, repeated to width.

    Stands in for the real encoder so no test downloads weights, and so the
    bytes in a cell can be anything: the stub never opens them as an image.
    """
    del model_name, device
    rows = [
        np.frombuffer(hashlib.sha256(payload).digest() * 12, dtype=np.uint8)
        for payload in payloads
    ]
    if not rows:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    return np.stack(rows)[:, :EMBEDDING_DIM].astype(np.float32)


@pytest.fixture
def stub_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(images_module, "encode_image_bytes", _stub_encoder)


def _b64(i: int) -> str:
    """A base64 cell; the stub encoder never opens the bytes, so any will do."""
    return base64.b64encode(f"image-{i}".encode()).decode("ascii")


def _png_bytes(
    colour: tuple[int, ...] | int = (255, 0, 0),
    size: tuple[int, int] = (8, 8),
    mode: str = "RGB",
) -> bytes:
    Image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    Image.new(mode, size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _frame(n: int = 40, cells: list | None = None) -> pd.DataFrame:
    """A numeric column beside `photo`: base64 cells unless `cells` says otherwise."""
    photo = [_b64(i) for i in range(n)] if cells is None else cells
    return pd.DataFrame({"num": np.arange(len(photo), dtype=float), "photo": photo})


def _expander(**kwargs: Any) -> ImageTransformer:
    return ImageTransformer(image_indices=[1], **kwargs)


def _estimator_data(
    estimator_cls: type, n: int = 120
) -> tuple[pd.DataFrame, np.ndarray]:
    """A numeric column beside `photo`, plus a `y` matching the estimator."""
    rng = np.random.default_rng(seed=42)
    X = pd.DataFrame(
        {"num": rng.normal(size=n), "photo": [_b64(i % 11) for i in range(n)]}
    )
    y = (
        rng.integers(0, 2, size=n)
        if estimator_cls is TabPFNClassifier
        else rng.normal(size=n)
    )
    return X, y


def _review_column(n: int) -> pd.Series:
    return pd.Series(
        [f"review {i}, a fairly long sentence" for i in range(n)], dtype="string"
    )


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


def _never_called(*_args: object, **_kwargs: object) -> np.ndarray:
    raise AssertionError("the encoder must not run when nothing is declared")


class TestDeclaration:
    """Which columns are expanded: the declared ones, and only when allowed."""

    def test__no_declared_columns__is_identity_and_calls_no_encoder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(images_module, "encode_image_bytes", _never_called)
        X = _frame()

        transformer = ImageTransformer()
        out = transformer.fit_transform(X)

        assert out is X
        assert transformer.expanded_indices == []
        assert transformer.feature_names_out_ == ["num", "photo"]

    @pytest.mark.usefixtures("stub_encoder")
    def test__declared_column__is_expanded_to_n_components(self) -> None:
        out = _expander().fit_transform(_frame())

        assert out.shape == (40, 1 + N_COMPONENTS)
        assert list(out.columns) == [
            "num",
            *[f"photo_img_{i}" for i in range(N_COMPONENTS)],
        ]
        assert out.notna().all().all()
        assert all(pd.api.types.is_float_dtype(dtype) for dtype in out.dtypes)

    def test__flag_off_with_declared_column__is_refused_naming_the_flag(self) -> None:
        with pytest.raises(TabPFNValidationError, match="TRANSFORM_IMAGE"):
            _expander(transform_image=False).fit(_frame())

    def test__flag_off_without_declared_columns__is_identity(self) -> None:
        X = _frame()

        assert ImageTransformer(transform_image=False).fit_transform(X) is X

    def test__declared_column_also_categorical__is_refused(self) -> None:
        with pytest.raises(
            TabPFNValidationError, match=r"categorical_features_indices.*1 \('photo'\)"
        ):
            _expander(categorical_indices=[1]).fit(_frame())

    def test__index_out_of_range__is_refused(self) -> None:
        with pytest.raises(TabPFNValidationError, match="has 2 columns"):
            ImageTransformer(image_indices=[5]).fit(_frame())

    def test__array_with_declared_columns__is_refused(self) -> None:
        with pytest.raises(TabPFNValidationError, match="DataFrame"):
            _expander().fit(np.zeros((3, 2)))

    def test__array_without_declared_columns__is_a_noop(self) -> None:
        X = np.zeros((3, 2))

        transformer = ImageTransformer().fit(X)

        assert transformer.transform(X) is X
        assert transformer.feature_names_out_ is None


@pytest.mark.usefixtures("stub_encoder")
class TestExpansion:
    """How a declared column is read and what it turns into."""

    def test__data_uri_prefix__is_stripped(self) -> None:
        plain = _expander().fit_transform(_frame())
        prefixed = _expander().fit_transform(
            _frame(cells=[f"data:image/png;base64,{_b64(i)}" for i in range(40)])
        )

        pd.testing.assert_frame_equal(prefixed, plain)

    def test__bytes_cells__are_accepted(self) -> None:
        plain = _expander().fit_transform(_frame())
        raw = _expander().fit_transform(
            _frame(cells=[f"image-{i}".encode() for i in range(40)])
        )

        pd.testing.assert_frame_equal(raw, plain)

    def test__whitespace_in_base64__is_tolerated(self) -> None:
        plain = _expander().fit_transform(_frame())
        wrapped = _expander().fit_transform(
            _frame(cells=[f" {_b64(i)[:4]}\n{_b64(i)[4:]} " for i in range(40)])
        )

        pd.testing.assert_frame_equal(wrapped, plain)

    def test__missing_cells__are_refused_naming_rows(self) -> None:
        cells: list[str | None] = [_b64(i) for i in range(40)]
        cells[3] = None
        cells[5] = ""

        with pytest.raises(TabPFNValidationError, match="no image in rows 3, 5"):
            _expander().fit(_frame(cells=cells))

    def test__non_base64_strings__are_refused_naming_rows(self) -> None:
        cells = [_b64(i) for i in range(40)]
        cells[2] = "not base64!!"

        with pytest.raises(TabPFNValidationError, match=r"rows 2: not base64"):
            _expander().fit(_frame(cells=cells))

    def test__fewer_rows_than_components__clamps_n_components(self) -> None:
        transformer = _expander()
        out = transformer.fit_transform(_frame(n=10))

        assert out.shape == (10, 1 + 10)
        assert transformer.fitted_columns_[1].pca.n_components_ == 10

    def test__declared_categorical_indices__are_remapped(self) -> None:
        X = pd.DataFrame({"photo": [_b64(i) for i in range(40)], "cat": ["a"] * 40})

        transformer = ImageTransformer(image_indices=[0], categorical_indices=[1])
        transformer.fit(X)

        assert transformer.output_indices([1]) == [0]
        assert transformer.output_indices(None) is None

    def test__generated_name_colliding_with_an_existing_column__is_deduped(
        self,
    ) -> None:
        X = _frame()
        X["photo_img_0"] = 1.0

        out = _expander().fit_transform(X)

        assert out.shape[1] == 2 + N_COMPONENTS
        assert len(set(out.columns)) == out.shape[1]
        assert list(out.columns[:2]) == ["num", "photo_img_0"]

    def test__duplicate_column_labels__are_handled_positionally(self) -> None:
        X = _frame().set_axis(["a", "a"], axis=1)

        out = _expander().fit_transform(X)

        assert out.shape == (40, 1 + N_COMPONENTS)
        assert out.columns[0] == "a"

    def test__caller_s_frame__is_left_untouched(self) -> None:
        X = _frame()
        before = X.copy()

        _expander().fit_transform(X)

        pd.testing.assert_frame_equal(X, before)

    def test__refit__replaces_the_last_fit(self) -> None:
        transformer = _expander()
        transformer.fit(_frame(n=40))

        out = transformer.fit_transform(_frame(n=10))

        assert out.shape == (10, 1 + 10)
        assert len(transformer.fitted_columns_[1].output_names) == 10


@pytest.mark.usefixtures("stub_encoder")
class TestExpansionAtPredictTime:
    """`transform` reapplies the fit and refuses what it cannot embed."""

    def test__same_data__reproduces_the_fitted_columns(self) -> None:
        X = _frame()
        transformer = _expander()
        fitted = transformer.fit_transform(X)

        pd.testing.assert_frame_equal(transformer.transform(X), fitted)

    def test__unseen_images__keep_the_fitted_width(self) -> None:
        transformer = _expander()
        fitted = transformer.fit_transform(_frame())

        out = transformer.transform(_frame(cells=[_b64(1000 + i) for i in range(7)]))

        assert list(out.columns) == list(fitted.columns)
        assert out.shape[0] == 7
        assert out.notna().all().all()

    def test__missing_cell_at_predict__is_refused_naming_the_row(self) -> None:
        transformer = _expander().fit(_frame())
        cells: list[str | None] = [_b64(i) for i in range(5)]
        cells[4] = None

        with pytest.raises(TabPFNValidationError, match="no image in rows 4"):
            transformer.transform(_frame(cells=cells))

    def test__undecodable_cell_at_predict__is_refused(self) -> None:
        transformer = _expander().fit(_frame())

        with pytest.raises(TabPFNValidationError, match="not base64"):
            transformer.transform(_frame(cells=["???"]))

    def test__other_embedding_width_at_predict__is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transformer = _expander().fit(_frame())
        monkeypatch.setattr(
            images_module,
            "encode_image_bytes",
            lambda payloads, **_: np.zeros((len(payloads), 16), dtype=np.float32),
        )

        with pytest.raises(TabPFNValidationError, match="16 dimensions now"):
            transformer.transform(_frame(n=3))

    def test__array_after_an_expanding_fit__is_refused(self) -> None:
        transformer = _expander().fit(_frame())

        with pytest.raises(TabPFNValidationError, match="DataFrame"):
            transformer.transform(np.zeros((3, 2)))

    def test__array_after_a_fit_that_expanded_nothing__passes(self) -> None:
        X = np.zeros((3, 2))

        assert ImageTransformer().fit(_frame()).transform(X) is X

    def test__device__is_handed_to_the_encoder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[torch.device] = []

        def recording(payloads: list[bytes], **kwargs: Any) -> np.ndarray:
            seen.append(kwargs["device"])
            return _stub_encoder(payloads, **kwargs)

        monkeypatch.setattr(images_module, "encode_image_bytes", recording)
        transformer = _expander(device="cpu").fit(_frame())
        transformer.device = torch.device("cpu", 0)
        transformer.transform(_frame(n=3))

        assert seen == [torch.device("cpu"), torch.device("cpu", 0)]


class TestInterface:
    """The parts shared with `DateTransformer` and `TextTransformer`."""

    @pytest.mark.usefixtures("stub_encoder")
    def test__fit__returns_itself_and_transform_matches_fit_transform(self) -> None:
        X = _frame()

        transformer = _expander().fit(X)

        assert isinstance(transformer, ImageTransformer)
        pd.testing.assert_frame_equal(
            transformer.transform(X), _expander().fit_transform(X)
        )

    def test__transform__before_fit__raises(self) -> None:
        with pytest.raises(NotFittedError):
            _expander().transform(_frame())
        with pytest.raises(NotFittedError):
            _ = _expander().expanded_indices

    def test__fit_on_an_array__has_no_output_names(self) -> None:
        assert ImageTransformer().fit(np.zeros((3, 2))).feature_names_out_ is None


class TestDecoding:
    """Reading the bytes of a cell into an RGB image, without any encoder."""

    def test__cell_to_bytes__reads_each_kind_of_cell(self) -> None:
        payload = b"image-0"
        encoded = base64.b64encode(payload).decode("ascii")

        assert images_module._cell_to_bytes(encoded) == payload
        assert images_module._cell_to_bytes(f"data:image/png;base64,{encoded}") == (
            payload
        )
        assert images_module._cell_to_bytes(f" {encoded[:3]}\n{encoded[3:]}") == (
            payload
        )
        assert images_module._cell_to_bytes(payload) == payload
        assert images_module._cell_to_bytes(bytearray(payload)) == payload
        for missing in (None, "", "  ", float("nan"), pd.NA):
            assert images_module._cell_to_bytes(missing) is None
        with pytest.raises(ValueError, match="not base64"):
            images_module._cell_to_bytes("not base64!!")
        with pytest.raises(ValueError, match="unsupported cell type int"):
            images_module._cell_to_bytes(5)

    def test__open_images__converts_every_mode_to_rgb_and_caps_the_size(
        self,
    ) -> None:
        payloads = [
            _png_bytes(),
            _png_bytes(colour=7, mode="L"),
            _png_bytes(colour=3, mode="P"),
            _png_bytes(size=(1000, 600)),
        ]

        images = images_module._open_images(payloads)

        assert [image.mode for image in images] == ["RGB"] * 4
        assert [image.size for image in images][:3] == [(8, 8)] * 3
        assert max(images[3].size) == images_module.MAX_IMAGE_SIDE

    def test__open_images__refuses_bytes_that_are_not_an_image(self) -> None:
        with pytest.raises(ValueError, match="row 1"):
            images_module._open_images([_png_bytes(), b"not an image"])


class TestEncoderLoading:
    """Importing the optional dependencies and fetching the weights."""

    def test__missing_optional_dependency__names_the_extra(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "torchvision", None)
        monkeypatch.setattr(images_module, "_ENCODERS", {})

        with pytest.raises(ImportError, match=r"tabpfn\[image\]"):
            images_module._get_image_encoder("some/model")

    def test__gated_repo__is_reported_with_license_instructions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class GatedAutoModel:
            @staticmethod
            def from_pretrained(name: str) -> None:
                raise OSError(f"You are trying to access a gated repo. {name}")

        monkeypatch.setattr(
            images_module,
            "_import_encoder_dependencies",
            lambda: (object(), GatedAutoModel),
        )
        monkeypatch.setattr(images_module, "_ENCODERS", {})

        with pytest.raises(TabPFNHuggingFaceGatedRepoError, match="some/model"):
            images_module._get_image_encoder("some/model")

    def test__encoder_cache__is_keyed_by_model_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loaded: list[str] = []

        class FakeModel:
            def eval(self) -> FakeModel:
                return self

        class FakeAuto:
            @staticmethod
            def from_pretrained(name: str) -> FakeModel:
                loaded.append(name)
                return FakeModel()

        monkeypatch.setattr(
            images_module, "_import_encoder_dependencies", lambda: (FakeAuto, FakeAuto)
        )
        monkeypatch.setattr(images_module, "_ENCODERS", {})

        first = images_module._get_image_encoder("a")
        images_module._get_image_encoder("b")
        again = images_module._get_image_encoder("a")

        assert again is first
        assert loaded == ["a", "a", "b", "b"]


@pytest.mark.usefixtures("stub_encoder")
@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_image_features_indices__expands_the_image_column(
    estimator_cls: type,
) -> None:
    """The declaration reaches the transformer, and the wider frame survives the
    whole fit/predict path: the schema describes the embedding features.
    """
    X, y = _estimator_data(estimator_cls)

    model = estimator_cls(n_estimators=1, device="cpu", image_features_indices=[1])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model.fit(X, y)

    schema = model.inferred_feature_schema_
    names = [
        feature.name.removeprefix(INPUT_FEATURE_PREFIX) for feature in schema.features
    ]
    assert names == ["num", *[f"photo_img_{i}" for i in range(N_COMPONENTS)]]
    assert schema.indices_for(FeatureModality.TEXT) == []
    assert model.image_transformer_.expanded_indices == [1]
    assert len(model.predict(X)) == len(X)


@pytest.mark.usefixtures("stub_encoder")
@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_image_n_components__sets_the_expanded_width(
    estimator_cls: type,
) -> None:
    X, y = _estimator_data(estimator_cls)

    model = estimator_cls(
        n_estimators=1,
        device="cpu",
        image_features_indices=[1],
        inference_config={"IMAGE_N_COMPONENTS": 5},
    ).fit(X, y)

    assert len(model.inferred_feature_schema_.features) == 1 + 5


@pytest.mark.usefixtures("stub_encoder")
@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_transform_image_off__refuses_the_declared_column(
    estimator_cls: type,
) -> None:
    X, y = _estimator_data(estimator_cls)

    model = estimator_cls(
        n_estimators=1,
        device="cpu",
        image_features_indices=[1],
        inference_config={"TRANSFORM_IMAGE": False},
    )

    with pytest.raises(TabPFNValidationError, match="TRANSFORM_IMAGE"):
        model.fit(X, y)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_without_image_features_indices__leaves_the_column_alone(
    estimator_cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is detected: an undeclared base64 column is read as before, and
    the encoder is never so much as imported.
    """
    monkeypatch.setattr(images_module, "encode_image_bytes", _never_called)
    X, y = _estimator_data(estimator_cls)

    model = estimator_cls(n_estimators=1, device="cpu").fit(X, y)

    assert model.image_transformer_.expanded_indices == []
    assert len(model.inferred_feature_schema_.features) == 2


@pytest.mark.usefixtures("stub_encoder")
@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_image_features_indices__reports_the_caller_s_own_columns(
    estimator_cls: type,
) -> None:
    """`n_features_in_` and `feature_names_in_` describe the frame the caller
    passed, not the expanded one, so an array of that width is refused at
    predict rather than silently misread.
    """
    X, y = _estimator_data(estimator_cls)

    model = estimator_cls(n_estimators=1, device="cpu", image_features_indices=[1])
    model.fit(X, y)

    assert model.n_features_in_ == 2
    assert list(model.feature_names_in_) == ["num", "photo"]
    assert len(model.inferred_feature_schema_.features) > 2
    with pytest.raises(TabPFNValidationError, match="DataFrame"):
        model.predict(np.zeros((10, 2)))


@pytest.mark.usefixtures("stub_encoder")
@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_image_and_categorical_overlap__is_refused(
    estimator_cls: type,
) -> None:
    X, y = _estimator_data(estimator_cls)

    model = estimator_cls(
        n_estimators=1,
        device="cpu",
        image_features_indices=[1],
        categorical_features_indices=[1],
    )

    with pytest.raises(TabPFNValidationError, match="categorical_features_indices"):
        model.fit(X, y)


@pytest.mark.usefixtures("stub_encoder")
@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_image_features_indices__stores_the_shifted_categorical_indices(
    estimator_cls: type,
) -> None:
    """The fitted attribute addresses the validated input, where the declared
    column has moved down past the expanded image column.
    """
    n = 80
    rng = np.random.default_rng(seed=0)
    X = pd.DataFrame(
        {
            "photo": [_b64(i) for i in range(n)],
            "cat": rng.choice(["a", "b", "c"], size=n),
            "num": rng.normal(size=n),
        }
    )
    y = rng.integers(0, 2, size=n) if estimator_cls is TabPFNClassifier else X["num"]

    model = estimator_cls(
        n_estimators=1,
        device="cpu",
        image_features_indices=[0],
        categorical_features_indices=[1],
    ).fit(X, y)
    assert model.categorical_features_indices_ == [0]

    model = estimator_cls(n_estimators=1, device="cpu", image_features_indices=[0]).fit(
        X, y
    )
    assert model.categorical_features_indices_ is None


@pytest.mark.usefixtures("stub_encoder")
@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_images_dates_and_text__declared_categorical_moves_past_all_three(
    estimator_cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared categorical column behind an expanded image, date and text
    column moves down once per expansion, since all three append their features
    at the end. The stored indices, the schema and the tuning estimators must
    all find it where it ends up, and the tuning estimators, fit on the expanded
    array, must not be told to embed anything.
    """
    n = 80
    rng = np.random.default_rng(seed=0)
    X = pd.DataFrame(
        {
            "photo": [_b64(i) for i in range(n)],
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
        image_features_indices=[0],
        categorical_features_indices=[3],
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
    assert all(
        name.startswith(("photo_img_", "when_", "review_")) for name in names[2:]
    )
    assert model.image_transformer_.expanded_indices == [0]
    # `when` sat at 1 and moved down once the image column ahead of it was
    # dropped; `review` sat at 2 and moved down twice.
    assert model.date_transformer_.expanded_indices == [0]
    assert model.text_transformer_.expanded_indices == [0]
    assert model.categorical_features_indices_ == [0]
    assert schema.indices_for(FeatureModality.CATEGORICAL) == [0]
    assert tuning_estimators
    assert all(
        estimator.categorical_features_indices == [0] for estimator in tuning_estimators
    )
    assert all(
        estimator.image_features_indices is None for estimator in tuning_estimators
    )
    assert len(model.predict(X)) == n


@pytest.mark.usefixtures("stub_encoder")
@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__predict_with_a_missing_image__is_refused_naming_the_row(
    estimator_cls: type,
) -> None:
    X, y = _estimator_data(estimator_cls)
    model = estimator_cls(n_estimators=1, device="cpu", image_features_indices=[1])
    model.fit(X, y)
    X_test = X.head(5).copy()
    X_test.loc[3, "photo"] = None

    with pytest.raises(TabPFNValidationError, match="no image in rows 3"):
        model.predict(X_test)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__fit_with_differentiable_input__sets_an_image_transformer(
    estimator_cls: type,
) -> None:
    """The differentiable path fits on a tensor, which holds no images, and still
    sets the attribute: every predict path converts through it unconditionally.
    Declaring image columns there is refused, since nothing could expand them.
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

    assert model.image_transformer_.transform(np.zeros((2, 3))).shape == (2, 3)

    declared = estimator_cls(
        n_estimators=1,
        device="cpu",
        differentiable_input=True,
        ignore_pretraining_limits=True,
        image_features_indices=[0],
    )
    with pytest.raises(ValueError, match="Image features are not supported"):
        declared.fit_with_differentiable_input(X, y)


@pytest.mark.usefixtures("stub_encoder")
@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test__save_and_load_fit_state__predicts_the_same_without_the_encoder(
    estimator_cls: type[TabPFNClassifier] | type[TabPFNRegressor], tmp_path: Path
) -> None:
    """The fitted transformer carries the scaler and PCA, never the encoder, so
    it pickles light; loading re-points its device to where the model lands.
    """
    X, y = _estimator_data(estimator_cls)
    model = estimator_cls(n_estimators=1, device="cpu", image_features_indices=[1])
    model.fit(X, y)
    expected = model.predict(X)

    path = tmp_path / "model.tabpfn_fit"
    model.save_fit_state(path)
    loaded = estimator_cls.load_from_fit_state(path, device="cpu")

    assert not any(
        isinstance(value, torch.nn.Module)
        for value in vars(loaded.image_transformer_).values()
    )
    assert loaded.image_transformer_.device == torch.device("cpu")
    np.testing.assert_allclose(loaded.predict(X), expected)


@pytest.mark.usefixtures("stub_encoder")
def test__predict_proba_batched_with_images__expands_the_test_frames() -> None:
    """The batched path validates each test frame the way `predict` does, so a
    declared image column has to be expanded there too, or the widths disagree.
    """
    X, y = _estimator_data(TabPFNClassifier)

    model = TabPFNClassifier(n_estimators=1, device="cpu", image_features_indices=[1])
    probabilities = model.predict_proba_batched([X, X], [y, y], [X, X])

    assert probabilities.shape[:2] == (2, len(X))


@pytest.mark.usefixtures("stub_encoder")
def test__get_embeddings_with_images__expands_the_frame() -> None:
    X, y = _estimator_data(TabPFNClassifier)
    model = TabPFNClassifier(n_estimators=1, device="cpu", image_features_indices=[1])
    model.fit(X, y)

    embeddings = model.get_embeddings(X)

    assert embeddings.shape[1] == len(X)


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
@pytest.mark.parametrize("indices", [["photo"], [-1], [1.5]])
def test__invalid_image_features_indices__are_refused(
    estimator_cls: type, indices: list
) -> None:
    X, y = _estimator_data(estimator_cls)

    model = estimator_cls(n_estimators=1, device="cpu", image_features_indices=indices)

    with pytest.raises(TabPFNValidationError, match="image_features_indices"):
        model.fit(X, y)


@pytest.mark.slow
def test__real_encoder__embeds_pngs_to_the_model_s_width() -> None:
    """Runs only with the optional dependencies and a Hub token whose account
    accepted the default encoder's license; skips everywhere else.
    """
    pytest.importorskip("transformers")
    pytest.importorskip("torchvision")
    pytest.importorskip("PIL")
    from huggingface_hub import get_token  # noqa: PLC0415

    if not (os.environ.get("HF_TOKEN") or get_token()):
        pytest.skip("needs a Hugging Face token with the encoder's license accepted")
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    payloads = [_png_bytes(colour) for colour in colours]

    embeddings = images_module.encode_image_bytes(
        payloads, model_name=DEFAULT_IMAGE_ENCODER_MODEL, device=torch.device("cpu")
    )

    assert embeddings.shape == (3, EMBEDDING_DIM)
    assert np.isfinite(embeddings).all()
    assert not np.allclose(embeddings[0], embeddings[1])
