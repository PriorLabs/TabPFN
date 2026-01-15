"""Encoder step to add frequency-based features to the input."""

from __future__ import annotations

from typing import Any

import torch

from tabpfn.architectures.encoders import SeqEncStep


class FrequencyFeatureEncoderStep(SeqEncStep):
    """Encoder step to add frequency-based features to the input."""

    def __init__(
        self,
        num_features: int,
        num_frequencies: int,
        freq_power_base: float = 2.0,
        max_wave_length: float = 4.0,
        **kwargs: Any,
    ):
        """Initialize the FrequencyFeatureEncoderStep.

        Args:
            num_features: The number of input features.
            num_frequencies: The number of frequencies to add (both sin and cos).
            freq_power_base:
                The base of the frequencies.
                Frequencies will be `freq_power_base`^i for i in range(num_frequencies).
            max_wave_length: The maximum wave length.
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.num_frequencies = num_frequencies
        self.num_features = num_features
        self.num_features_out = num_features + 2 * num_frequencies * num_features
        self.freq_power_base = freq_power_base
        # We add frequencies with a factor of freq_power_base
        wave_lengths = torch.tensor(
            [freq_power_base**i for i in range(num_frequencies)],
            dtype=torch.float,
        )
        wave_lengths = wave_lengths / wave_lengths[-1] * max_wave_length
        # After this adaption, the last (highest) wavelength is max_wave_length
        self.register_buffer("wave_lengths", wave_lengths)

    def _fit(
        self,
        x: torch.Tensor,
        single_eval_pos: int | None = None,
        categorical_inds: list[int] | None = None,
    ) -> None:
        """Fit the encoder step. Does nothing for FrequencyFeatureEncoderStep."""

    def _transform(
        self,
        x: torch.Tensor,
        single_eval_pos: int | None = None,  # noqa: ARG002
        categorical_inds: list[int] | None = None,  # noqa: ARG002
    ) -> tuple[torch.Tensor]:
        """Add frequency-based features to the input tensor.

        Args:
            x: The input tensor of shape (seq_len, batch_size, num_features).
            single_eval_pos: The position to use for single evaluation. Not used.
            categorical_inds: The indices of categorical features. Not used.

        Returns:
            A tuple containing the transformed tensor of shape
            `(seq_len, batch_size, num_features + 2 * num_frequencies * num_features)`.
        """
        extended = x[..., None] / self.wave_lengths[None, None, None, :] * 2 * torch.pi
        new_features = torch.cat(
            (x[..., None], torch.sin(extended), torch.cos(extended)),
            -1,
        )
        new_features = new_features.reshape(*x.shape[:-1], -1)
        return (new_features,)
