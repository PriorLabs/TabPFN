"""Encoder step to normalize the input in different ways."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tabpfn.architectures.encoders import SeqEncStep

from ._ops import (
    normalize_data,
    remove_outliers,
)

if TYPE_CHECKING:
    import torch


class InputNormalizationEncoderStep(SeqEncStep):
    """Encoder step to normalize the input in different ways.

    Can be used to normalize the input to a ranking, remove outliers,
    or normalize the input to have unit variance.
    """

    def __init__(
        self,
        *,
        normalize_on_train_only: bool,
        normalize_to_ranking: bool,
        normalize_x: bool,
        remove_outliers: bool,
        remove_outliers_sigma: float = 4.0,
        seed: int = 0,
        **kwargs: Any,
    ):
        """Initialize the InputNormalizationEncoderStep.

        Args:
            normalize_on_train_only: Whether to compute normalization only on the
            training set.
            normalize_to_ranking: Whether to normalize the input to a ranking.
            normalize_x: Whether to normalize the input to have unit variance.
            remove_outliers: Whether to remove outliers from the input.
            remove_outliers_sigma: The number of standard deviations to use for outlier
            removal.
            seed: Random seed for reproducibility.
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.normalize_on_train_only = normalize_on_train_only
        self.normalize_to_ranking = normalize_to_ranking
        self.normalize_x = normalize_x
        self.remove_outliers = remove_outliers
        self.remove_outliers_sigma = remove_outliers_sigma
        self.seed = seed
        self.reset_seed()
        self.register_buffer("lower_for_outlier_removal", None, persistent=False)
        self.register_buffer("upper_for_outlier_removal", None, persistent=False)
        self.register_buffer("mean_for_normalization", None, persistent=False)
        self.register_buffer("std_for_normalization", None, persistent=False)

    def reset_seed(self) -> None:
        """Reset the random seed."""

    def _fit(self, x: torch.Tensor, single_eval_pos: int, **kwargs: Any) -> None:  # noqa: ARG002
        """Compute the normalization statistics on the training set.

        Args:
            x: The input tensor.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments (unused).
        """
        normalize_position = single_eval_pos if self.normalize_on_train_only else -1
        if self.remove_outliers and not self.normalize_to_ranking:
            (
                x,
                (
                    self.lower_for_outlier_removal,
                    self.upper_for_outlier_removal,
                ),
            ) = remove_outliers(
                x,
                normalize_positions=normalize_position,
                n_sigma=self.remove_outliers_sigma,
            )

        if self.normalize_x:
            (
                x,
                (
                    self.mean_for_normalization,
                    self.std_for_normalization,
                ),
            ) = normalize_data(
                x,
                normalize_positions=normalize_position,
                return_scaling=True,
            )

    def _transform(
        self,
        x: torch.Tensor,
        single_eval_pos: int,
        **kwargs: Any,  # noqa: ARG002
    ) -> tuple[torch.Tensor]:
        """Normalize the input tensor.

        Args:
            x: The input tensor.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the normalized tensor.
        """
        normalize_position = single_eval_pos if self.normalize_on_train_only else -1

        if self.normalize_to_ranking:
            raise AssertionError(
                "Not implemented currently as it was not used in a long time and hard "
                "to move out the state.",
            )

        if self.remove_outliers:
            assert self.remove_outliers_sigma > 1.0, (
                "remove_outliers_sigma must be > 1.0"
            )

            x, _ = remove_outliers(
                x,
                normalize_positions=normalize_position,
                lower=self.lower_for_outlier_removal,
                upper=self.upper_for_outlier_removal,
                n_sigma=self.remove_outliers_sigma,
            )

        if self.normalize_x:
            x = normalize_data(
                x,
                normalize_positions=normalize_position,
                mean=self.mean_for_normalization,
                std=self.std_for_normalization,
            )
        return (x,)
