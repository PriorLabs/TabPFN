"""Encoder step to add frequency-based features to the input."""

from __future__ import annotations

from typing import Any
from typing_extensions import override

import torch

from tabpfn.architectures.encoders import TorchPreprocessingStep


class FrequencyFeatureEncoderStep(TorchPreprocessingStep):
    """Encoder step to add frequency-based features to the input."""

    def __init__(
        self,
        num_features: int,
        num_frequencies: int,
        freq_power_base: float = 2.0,
        max_wave_length: float = 4.0,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main",),
    ):
        """Initialize the FrequencyFeatureEncoderStep.

        Args:
            num_features: The number of input features.
            num_frequencies: The number of frequencies to add (both sin and cos).
            freq_power_base:
                The base of the frequencies.
                Frequencies will be `freq_power_base`^i for i in range(num_frequencies).
            max_wave_length: The maximum wave length.
            in_keys: The keys of the input tensors.
            out_keys: The keys to assign the output tensors to.
        """
        assert len(in_keys) == len(out_keys) == 1, (
            f"{self.__class__.__name__} expects a single input and output key."
        )

        super().__init__(in_keys, out_keys)
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

    @override
    def _fit(
        self,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> None:
        """Fit the encoder step. Does nothing for FrequencyFeatureEncoderStep."""
        del state, kwargs

    @override
    def _transform(
        self,
        state: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Add frequency-based features to the input tensor.

        Args:
            x: The input tensor of shape (seq_len, batch_size, num_features).
            single_eval_pos: The position to use for single evaluation. Not used.
            categorical_inds: The indices of categorical features. Not used.

        Returns:
            A dict mapping `out_keys[0]` to the transformed tensor of shape
            `(seq_len, batch_size, num_features + 2 * num_frequencies * num_features)`.
        """
        del kwargs
        x = state[self.in_keys[0]]
        extended = x[..., None] / self.wave_lengths[None, None, None, :] * 2 * torch.pi
        new_features = torch.cat(
            (x[..., None], torch.sin(extended), torch.cos(extended)),
            -1,
        )
        new_features = new_features.reshape(*x.shape[:-1], -1)
        return {self.out_keys[0]: new_features}
