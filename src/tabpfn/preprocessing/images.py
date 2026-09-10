#  Copyright (c) Prior Labs GmbH 2026.

"""Expand a DataFrame's declared image columns into numeric features before validation.

An image column is one the caller names in `image_features_indices`: nothing is
detected, so an undeclared column is never touched and, with none declared, this
is an identity that imports none of the optional image dependencies. Each cell of
a declared column holds one image, as a base64 string (a `data:image/...;base64,`
prefix is tolerated) or as the image file's `bytes`. With `TRANSFORM_IMAGE` on,
the column is replaced by `IMAGE_N_COMPONENTS` numeric features: the CLS embedding
of a vision transformer (`IMAGE_ENCODER_MODEL`, a DINOv3 ViT-S/16 by default),
standardised and reduced by a PCA fit on the training rows. Fewer features when
the column has fewer rows than that. With the flag off, a declared column is
refused with an error naming the flag.

At predict, the same columns are embedded by the same encoder and projected with
the PCA fit at training time, so an unseen image lands where its training
neighbours are. A cell that is missing, or that cannot be decoded as an image, is
refused rather than guessed at, at fit and at predict alike.

The encoder is downloaded from the Hugging Face Hub on first use and kept in a
module-level cache for the process. It is never stored on the transformer, so a
fitted estimator pickles without it. The default weights are gated: accept their
license on the Hub once and log in (`hf auth login` or `HF_TOKEN`). The optional
dependencies come with `pip install "tabpfn[image]"`.

Only `TabPFNClassifier` and `TabPFNRegressor` run this, first, before
`DateTransformer` and `TextTransformer`. The fine-tuning estimators validate their
input directly.

Column handling is positional throughout: labels are the caller's and can repeat.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import io
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from tabpfn.errors import TabPFNHuggingFaceGatedRepoError, TabPFNValidationError
from tabpfn.preprocessing.datamodel import make_names_unique

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tabpfn.constants import XType

__all__ = ["ImageTransformer"]

DEFAULT_IMAGE_ENCODER_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
IMAGE_BATCH_SIZE = 64
MAX_IMAGE_SIDE = 512
"""Longest side an image is shrunk to before the encoder's own resize. Caps memory
on high-resolution inputs with no effect on the embedding."""

_ENCODERS: dict[str, tuple[Any, Any]] = {}
"""Loaded (model, processor) per Hugging Face model id, for the life of the process.
Kept out of the transformer so a fitted estimator pickles without the weights."""


@dataclasses.dataclass
class _FittedImageColumn:
    """One input column's fitted reduction and its features' names."""

    scaler: StandardScaler
    pca: PCA
    output_names: list[str]
    embedding_dim: int


class ImageTransformer:
    """Expands each declared image column into numeric features, when asked to.

    Used like `TextTransformer`, and right before `DateTransformer`: `fit_transform`
    once at fit time, keep the instance as `image_transformer_`, `transform` at
    predict time.

    Args:
        image_indices: Positions in the `X` handed here whose cells hold images.
            `None` or empty: nothing is expanded and nothing optional is imported.
        categorical_indices: Indices the caller declared categorical. An image
            column among them is refused: it becomes many numeric columns, so
            there is no single column the declaration could apply to.
        transform_image: Whether a declared image column is embedded and reduced.
            Off, a declared column is refused with an error naming the flag.
        n_components: Features an image column is expanded into, at most: fewer
            when the column has fewer rows, or the embedding fewer dimensions.
        model_name: Hugging Face id of the encoder; its CLS token is the embedding.
        device: Where the encoder runs. `None` for the CPU. A public attribute:
            the estimator re-points it when it moves devices or is loaded.

    Attributes:
        fitted_columns_: Input position -> the scaler and PCA fit on that column's
            embeddings and the names of the features they make. Empty when
            nothing was expanded.
        feature_names_out_: The transformed frame's column labels as strings, or
            `None` when the input was not a `DataFrame` and so has no labels.
    """

    fitted_columns_: dict[int, _FittedImageColumn]
    feature_names_out_: list[str] | None

    def __init__(
        self,
        *,
        image_indices: Sequence[int] | None = None,
        categorical_indices: Sequence[int] | None = None,
        transform_image: bool = True,
        n_components: int = 30,
        model_name: str = DEFAULT_IMAGE_ENCODER_MODEL,
        device: torch.device | str | None = None,
    ) -> None:
        self._image_indices = sorted(set(image_indices or ()))
        self._declared_categorical = set(categorical_indices or ())
        self._transform_image = transform_image
        self._n_components = n_components
        self._model_name = model_name
        self.device = torch.device("cpu") if device is None else torch.device(device)

    @property
    def expanded_indices(self) -> list[int]:
        """Input positions that were expanded into image features, ascending."""
        self._check_is_fitted()
        return sorted(self.fitted_columns_)

    def fit(self, X: XType) -> ImageTransformer:
        """Fit one reduction per declared image column in `X`.

        Args:
            X: The input data, before any dtype fixing.

        Returns:
            Itself, fitted.
        """
        self._fit(X)
        return self

    def fit_transform(self, X: XType) -> XType:
        """`fit(X).transform(X)`, embedding each image column only once.

        Expansion changes the column count: `feature_names_out_` reports the
        resulting labels and `output_indices` moves input indices to match.

        Args:
            X: The input data, before any dtype fixing.

        Returns:
            `X`, converted as `transform` would.
        """
        blocks = self._fit(X)
        if not isinstance(X, pd.DataFrame) or not blocks:
            return X
        return _drop_and_append(X.reset_index(drop=True), self.expanded_indices, blocks)

    def transform(self, X: XType) -> XType:
        """Reapply the expansion `fit` decided on, so the width holds.

        Args:
            X: The data, before any dtype fixing.

        Raises:
            NotFittedError: If `fit` has not run yet.
            TabPFNValidationError: If a cell of an expanded column is missing or
                not an image, if the encoder embeds to another width than at fit,
                or if `fit` expanded columns and `X` is not a `DataFrame`, the
                only input that can carry them.
        """
        self._check_is_fitted()
        if not isinstance(X, pd.DataFrame):
            _refuse_array_after_expansion(X, self.expanded_indices)
            return X
        if not self.expanded_indices:
            return X
        blocks = [
            self._apply_one(X, i, self.fitted_columns_[i])
            for i in self.expanded_indices
        ]
        return _drop_and_append(X.reset_index(drop=True), self.expanded_indices, blocks)

    def output_indices(self, indices: Sequence[int] | None) -> list[int] | None:
        """Where each of `indices`, input positions, sits in the transformed frame.

        A kept column shifts down by however many expanded columns sat ahead of
        it. An expanded position is never asked for: the one caller passes
        declared-categorical indices, which are refused as image columns.

        Args:
            indices: Input positions, or `None` for none declared.

        Returns:
            The same positions in the transformed frame, or `None` for `None`.
        """
        self._check_is_fitted()
        if indices is None:
            return None
        expanded = self.expanded_indices
        return [i - sum(1 for j in expanded if j < i) for i in indices]

    def _check_is_fitted(self) -> None:
        # By hand: sklearn's `check_is_fitted` requires a `BaseEstimator`.
        if not hasattr(self, "fitted_columns_"):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. Call "
                "`fit` before using `transform`."
            )

    def _fit(self, X: XType) -> list[pd.DataFrame]:
        """Fit the reductions; return each expanded column's features, in order."""
        # Cleared first, so refitting on an input with nothing to expand still
        # forgets the last fit.
        self.fitted_columns_ = {}
        self.feature_names_out_ = None
        positions = self._image_indices
        if not positions:
            if isinstance(X, pd.DataFrame):
                self.feature_names_out_ = [str(column) for column in X.columns]
            return []
        if not isinstance(X, pd.DataFrame):
            _refuse_non_dataframe(X, positions)
            return []
        _refuse_out_of_range(X, [i for i in positions if i >= X.shape[1]])
        if not self._transform_image:
            _refuse_flag_off(X, positions)
        _refuse_declared_categorical(
            X, [i for i in positions if i in self._declared_categorical]
        )

        kept_names = [
            str(column) for i, column in enumerate(X.columns) if i not in set(positions)
        ]
        expanded_names: list[str] = []
        blocks: list[pd.DataFrame] = []
        for position in positions:
            block, fitted = self._fit_one(
                self._embed(X, position),
                str(X.columns[position]),
                kept_names + expanded_names,
                self._n_components,
            )
            self.fitted_columns_[position] = fitted
            expanded_names += fitted.output_names
            blocks.append(block)
        self.feature_names_out_ = kept_names + expanded_names
        return blocks

    def _embed(self, X: pd.DataFrame, position: int) -> np.ndarray:
        """Embed every cell of column `position`, refusing missing or broken ones."""
        payloads = _payloads(X, position)
        try:
            return encode_image_bytes(
                payloads, model_name=self._model_name, device=self.device
            )
        except ValueError as e:
            raise TabPFNValidationError(
                f"Column {_name_columns(X, [position])} holds a cell that is not an "
                f"image ({e}). Pass each image as a base64-encoded string or as the "
                "image file's bytes."
            ) from e

    @staticmethod
    def _fit_one(
        embeddings: np.ndarray,
        label: str,
        existing_names: Sequence[str],
        n_components: int,
    ) -> tuple[pd.DataFrame, _FittedImageColumn]:
        """Fit a scaler and a PCA on one column's embeddings, named after the column.

        A PCA keeps at most as many components as it has rows or dimensions, so
        how many features there are is settled here, which is why `transform`
        reuses this fit rather than making a fresh one.
        """
        n_kept = min(n_components, *embeddings.shape)
        scaler = StandardScaler().fit(embeddings)
        scaled = scaler.transform(embeddings)
        # Seeded on its own: the features a column turns into are a property of
        # the data, and should not move with the estimator's seed.
        pca = PCA(n_components=n_kept, random_state=0).fit(scaled)
        reduced = pca.transform(scaled).astype(np.float32)
        output_names = make_names_unique(
            [f"{label}_img_{i}" for i in range(n_kept)], existing=existing_names
        )
        return (
            pd.DataFrame(reduced, columns=output_names),
            _FittedImageColumn(
                scaler=scaler,
                pca=pca,
                output_names=output_names,
                embedding_dim=embeddings.shape[1],
            ),
        )

    def _apply_one(
        self, X: pd.DataFrame, position: int, fitted: _FittedImageColumn
    ) -> pd.DataFrame:
        """Reapply one fitted reduction, naming its features as at fit."""
        embeddings = self._embed(X, position)
        _refuse_other_embedding_dim(
            X, position, fit_dim=fitted.embedding_dim, now_dim=embeddings.shape[1]
        )
        reduced = fitted.pca.transform(fitted.scaler.transform(embeddings))
        return pd.DataFrame(reduced.astype(np.float32), columns=fitted.output_names)


def _cell_to_bytes(cell: object) -> bytes | None:
    """The image bytes one cell holds, or `None` for a missing cell.

    A `str` is base64, optionally behind a `data:...;base64,` prefix and with
    whitespace removed; `bytes` are the image file itself.

    Raises:
        ValueError: On a value that is neither, or a string that is not base64.
    """
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return bytes(cell)
    if isinstance(cell, str):
        text = cell.strip()
        if not text:
            return None
        if text.startswith("data:"):
            text = text.split(",", 1)[-1]
        try:
            return base64.b64decode("".join(text.split()), validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError(f"not base64: {e}") from e
    if cell is None or (pd.api.types.is_scalar(cell) and pd.isna(cell)):
        return None
    raise ValueError(f"unsupported cell type {type(cell).__name__}")


def _payloads(X: pd.DataFrame, position: int) -> list[bytes]:
    """Every cell of column `position` as image bytes, in row order.

    Raises:
        TabPFNValidationError: Naming the rows that are missing or not base64.
    """
    payloads: list[bytes] = []
    missing: list[int] = []
    bad: dict[int, str] = {}
    for row, cell in enumerate(X.iloc[:, position].tolist()):
        try:
            payload = _cell_to_bytes(cell)
        except ValueError as e:
            bad[row] = str(e)
            continue
        if payload is None:
            missing.append(row)
        else:
            payloads.append(payload)
    _refuse_missing_images(X, position, missing)
    _refuse_undecodable_cells(X, position, bad)
    return payloads


def _import_pil() -> Any:
    """PIL's `Image` module, or an `ImportError` naming the extra that brings it."""
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "Image columns need TabPFN's optional image dependencies: "
            'pip install "tabpfn[image]".'
        ) from e
    return Image


def _import_encoder_dependencies() -> tuple[Any, Any]:
    """Transformers' auto classes, or an `ImportError` naming the extra."""
    try:
        import torchvision  # noqa: F401, PLC0415
        from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "Image columns need TabPFN's optional image dependencies (transformers, "
            'torchvision, pillow): pip install "tabpfn[image]".'
        ) from e
    return AutoImageProcessor, AutoModel


def _get_image_encoder(model_name: str) -> tuple[Any, Any]:
    """The encoder and its processor for `model_name`, loaded once per process.

    Raises:
        TabPFNHuggingFaceGatedRepoError: When the weights are gated and the
            license has not been accepted, or no token is available.
    """
    if model_name not in _ENCODERS:
        AutoImageProcessor, AutoModel = _import_encoder_dependencies()
        try:
            model = AutoModel.from_pretrained(model_name).eval()
            processor = AutoImageProcessor.from_pretrained(model_name)
        except OSError as e:
            # transformers folds the Hub's `GatedRepoError` into an `OSError`.
            if "gated" in str(e).lower():
                raise TabPFNHuggingFaceGatedRepoError(model_name) from e
            raise
        _ENCODERS[model_name] = (model, processor)
    return _ENCODERS[model_name]


def _open_images(payloads: Sequence[bytes]) -> list[Any]:
    """Decode each payload to an RGB PIL image no larger than `MAX_IMAGE_SIDE`.

    A palette image goes through RGBA so its transparency survives the
    conversion; any other mode, grayscale included, converts to RGB directly.

    Raises:
        ValueError: Naming the row whose bytes PIL cannot read as an image.
    """
    Image = _import_pil()
    images = []
    for row, payload in enumerate(payloads):
        try:
            image = Image.open(io.BytesIO(payload))
            image.load()
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as e:
            raise ValueError(f"row {row}: {e}") from e
        if image.mode == "P":
            image = image.convert("RGBA")
        image = image.convert("RGB")
        if max(image.size) > MAX_IMAGE_SIDE:
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.LANCZOS)
        images.append(image)
    return images


def encode_image_bytes(
    payloads: Sequence[bytes], *, model_name: str, device: torch.device
) -> np.ndarray:
    """The CLS embedding of every image, as `(len(payloads), hidden_size)` float32.

    Batches of `IMAGE_BATCH_SIZE`, no gradients. The encoder is moved to `device`
    in place, a no-op once it is there.

    Args:
        payloads: The image files' bytes, one per row.
        model_name: Hugging Face id of the encoder.
        device: Where the encoder runs.

    Raises:
        ValueError: Naming the row whose bytes are not an image.
    """
    images = _open_images(payloads)
    model, processor = _get_image_encoder(model_name)
    model.to(device)
    chunks = []
    for start in range(0, len(images), IMAGE_BATCH_SIZE):
        batch = images[start : start + IMAGE_BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state
        chunks.append(hidden[:, 0, :].float().cpu().numpy())
    if not chunks:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _drop_and_append(
    frame: pd.DataFrame, expanded: Sequence[int], blocks: Sequence[pd.DataFrame]
) -> pd.DataFrame:
    """Drop the `expanded` positions and append `blocks` after the kept columns.

    The blocks are default-indexed, so the kept columns must be too, or `concat`
    aligns the two by label instead of position.
    """
    keep = [i for i in range(frame.shape[1]) if i not in set(expanded)]
    return pd.concat([frame.iloc[:, keep], *blocks], axis=1)


def _name_columns(X: pd.DataFrame, positions: Sequence[int]) -> str:
    """Name each of `positions` by index and label, e.g. `1 ('photo')`."""
    return ", ".join(f"{i} ({X.columns[i]!r})" for i in positions)


def _name_rows(rows: Sequence[int], limit: int = 10) -> str:
    """List the first `limit` of `rows`, then say how many more there are."""
    shown = ", ".join(str(row) for row in rows[:limit])
    if len(rows) > limit:
        shown += f" and {len(rows) - limit} more"
    return shown


def _refuse_flag_off(X: pd.DataFrame, positions: Sequence[int]) -> None:
    """Raise on the declared image columns at `positions`, the flag being off."""
    raise TabPFNValidationError(
        f"These columns are listed in `image_features_indices` but `TRANSFORM_IMAGE` "
        f"is off: {_name_columns(X, positions)}. Set "
        '`inference_config={"TRANSFORM_IMAGE": True}` to embed them, or drop them '
        "from `image_features_indices`."
    )


def _refuse_non_dataframe(X: XType, positions: Sequence[int]) -> None:
    """Raise on a fit input that is not a `DataFrame` yet declares image columns."""
    raise TabPFNValidationError(
        f"`image_features_indices` names columns {list(positions)}, but only a "
        f"DataFrame can carry images; got {type(X).__name__}. Pass `X` as a "
        "DataFrame whose declared columns hold each image as a base64 string or "
        "as bytes."
    )


def _refuse_out_of_range(X: pd.DataFrame, positions: Sequence[int]) -> None:
    """Raise on declared `positions` that `X` does not have."""
    if not positions:
        return
    raise TabPFNValidationError(
        f"`image_features_indices` names positions {list(positions)}, but `X` has "
        f"{X.shape[1]} columns."
    )


def _refuse_declared_categorical(X: pd.DataFrame, positions: Sequence[int]) -> None:
    """Raise on the image columns at `positions`, all declared categorical too."""
    if not positions:
        return
    raise TabPFNValidationError(
        f"These columns are listed in both `image_features_indices` and "
        f"`categorical_features_indices`: {_name_columns(X, positions)}. An image "
        "column is expanded into numeric features, so it cannot be a category as "
        "well: drop it from one of the two."
    )


def _refuse_missing_images(X: pd.DataFrame, position: int, rows: Sequence[int]) -> None:
    """Raise on the cells of column `position` at `rows`, which hold no image."""
    if not rows:
        return
    raise TabPFNValidationError(
        f"Column {_name_columns(X, [position])} has no image in rows "
        f"{_name_rows(rows)}. Every cell of a declared image column has to hold a "
        "base64 string or bytes: drop those rows or fill the image in first."
    )


def _refuse_undecodable_cells(
    X: pd.DataFrame, position: int, bad: dict[int, str]
) -> None:
    """Raise on the cells of column `position` in `bad`, `{row: reason}`."""
    if not bad:
        return
    first_reason = next(iter(bad.values()))
    raise TabPFNValidationError(
        f"Column {_name_columns(X, [position])} holds cells that are not base64 "
        f"images in rows {_name_rows(list(bad))}: {first_reason}. Pass each image "
        "as a base64-encoded string (a `data:image/...;base64,` prefix is fine) or "
        "as the image file's bytes."
    )


def _refuse_other_embedding_dim(
    X: pd.DataFrame, position: int, *, fit_dim: int, now_dim: int
) -> None:
    """Raise when column `position` embeds to another width than when `fit` ran."""
    if fit_dim == now_dim:
        return
    raise TabPFNValidationError(
        f"Column {_name_columns(X, [position])} embeds to {now_dim} dimensions now "
        f"but did to {fit_dim} when `fit` ran, so the encoder changed. Use the same "
        "`IMAGE_ENCODER_MODEL` as at fit."
    )


def _refuse_array_after_expansion(X: XType, expanded: Sequence[int]) -> None:
    """Raise on a non-`DataFrame` predict input once `fit` expanded columns.

    Its raw width may match `n_features_in_`, so the shape check upstream let it
    through, but nothing here can widen an array to the expanded layout.
    """
    if not expanded:
        return
    raise TabPFNValidationError(
        f"`fit` expanded the image columns at positions {list(expanded)} into "
        "numeric features, so predict input has to be a DataFrame carrying those "
        f"columns as base64 strings or bytes; got {type(X).__name__}."
    )
