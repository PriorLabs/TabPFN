"""Label encoding utilities for TabPFN classifier."""

from __future__ import annotations

import dataclasses
import typing

import numpy as np
from sklearn.preprocessing import LabelEncoder

from tabpfn.validation import validate_num_classes


@dataclasses.dataclass
class LabelMetadata:
    """Result of label encoding containing transformed labels and metadata."""

    classes: np.ndarray
    n_classes: int
    class_counts: np.ndarray


class TabPFNLabelEncoder:
    """Handles label encoding and validation for TabPFN classifier.

    Wraps sklearn's LabelEncoder and adds validation for maximum number of classes,
    as well as computing class counts. Also keeps the target name as a string.
    """

    def __init__(self, original_target_name: str | None = None) -> None:
        super().__init__()
        self._encoder = LabelEncoder()
        self.original_target_name = original_target_name

    def fit_transform(
        self,
        y: np.ndarray,
        max_num_classes: int,
    ) -> tuple[np.ndarray, LabelMetadata]:
        """Fit and transform labels, returning encoded labels and metadata.

        Args:
            y: Target labels to encode.
            max_num_classes: Maximum number of classes allowed.

        Returns:
            LabelEncodingResult containing transformed labels and class metadata.
        """
        y_encoded = self._encoder.fit_transform(y)
        classes = typing.cast("np.ndarray", self._encoder.classes_)
        n_classes = len(classes)

        _, counts = np.unique(y_encoded, return_counts=True)

        validate_num_classes(
            num_classes=n_classes,
            max_num_classes=max_num_classes,
        )

        return typing.cast("np.ndarray", y_encoded), LabelMetadata(
            classes=classes,
            n_classes=n_classes,
            class_counts=counts,
        )

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform encoded labels back to original labels.

        Args:
            y: Encoded labels to transform back.

        Returns:
            Original labels before encoding.
        """
        return self._encoder.inverse_transform(y)
