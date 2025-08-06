#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

from typing import Any, Literal, overload

import numpy as np
import torch
from torch import nn


# usage of custom implementations is required to support ONNX export
def torch_nansum(
    x: torch.Tensor,
    axis: int | tuple[int, ...] | None = None,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Computes the sum of a tensor, treating NaNs as zero.

    Args:
        x: The input tensor.
        axis: The dimension or dimensions to reduce.
        keepdim: Whether the output tensor has `axis` retained or not.
        dtype: The desired data type of the returned tensor.

    Returns:
        The sum of the tensor with NaNs treated as zero.
    """
    nan_mask = torch.isnan(x)
    masked_input = torch.where(
        nan_mask,
        torch.tensor(0.0, device=x.device, dtype=x.dtype),
        x,
    )
    return torch.sum(masked_input, axis=axis, keepdim=keepdim, dtype=dtype)


@overload
def torch_nanmean(
    x: torch.Tensor,
    axis: int = 0,
    *,
    return_nanshare: Literal[False] = False,
    include_inf: bool = False,
) -> torch.Tensor: ...


@overload
def torch_nanmean(
    x: torch.Tensor,
    axis: int = 0,
    *,
    return_nanshare: Literal[True],
    include_inf: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]: ...


def torch_nanmean(
    x: torch.Tensor,
    axis: int = 0,
    *,
    return_nanshare: bool = False,
    include_inf: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Computes the mean of a tensor over a given dimension, ignoring NaNs.

    Designed for stability: If all inputs are NaN, the mean will be 0.0.

    Args:
        x: The input tensor.
        axis: The dimension to reduce.
        return_nanshare: If True, also return the proportion of NaNs.
        include_inf: If True, treat infinity as NaN for the purpose of the calculation.

    Returns:
        The mean of the input tensor, ignoring NaNs. If `return_nanshare` is True,
        returns a tuple containing the mean and the share of NaNs.
    """
    nan_mask = torch.isnan(x)
    if include_inf:
        nan_mask = torch.logical_or(nan_mask, torch.isinf(x))

    num = torch.where(nan_mask, torch.full_like(x, 0), torch.full_like(x, 1)).sum(  # type: ignore
        axis=axis,
    )
    value = torch.where(nan_mask, torch.full_like(x, 0), x).sum(axis=axis)  # type: ignore
    if return_nanshare:
        return value / num, 1.0 - (num / x.shape[axis])
    return value / num.clip(min=1.0)


def torch_nanstd(x: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Computes the standard deviation of a tensor over a given dimension, ignoring NaNs.

    This implementation is designed for stability. It clips the denominator `(num - 1)`
    at a minimum of 1. This prevents division-by-zero errors that would produce `NaN`
    or `inf` when calculating the standard deviation of a single data point.

    Args:
        x: The input tensor.
        axis: The dimension to reduce.

    Returns:
        The standard deviation of the input tensor, ignoring NaNs.
    """
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(  # type: ignore
        axis=axis,
    )
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)  # type: ignore
    mean = value / num.clip(min=1.0)
    mean_broadcast = torch.repeat_interleave(
        mean.unsqueeze(axis),
        x.shape[axis],
        dim=axis,
    )
    # Clip the denominator to avoid division by zero when num=1
    var = torch_nansum(torch.square(mean_broadcast - x), axis=axis) / (num - 1).clip(
        min=1.0
    )
    return torch.sqrt(var)


@overload
def normalize_data(
    data: torch.Tensor,
    *,
    normalize_positions: int = -1,
    return_scaling: Literal[False] = False,
    clip: bool = True,
    std_only: bool = False,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor: ...


@overload
def normalize_data(
    data: torch.Tensor,
    *,
    normalize_positions: int = -1,
    return_scaling: Literal[True],
    clip: bool = True,
    std_only: bool = False,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: ...


def normalize_data(
    data: torch.Tensor,
    *,
    normalize_positions: int = -1,
    return_scaling: bool = False,
    clip: bool = True,
    std_only: bool = False,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Normalize data to mean 0 and std 1 with high numerical stability.

    This function is designed to be robust against several edge cases:
    1.  **Constant Features**: If a feature is constant, its standard deviation (`std`)
        will be 0. This is handled by replacing `std=0` with `1` to prevent `0/0`
        division, effectively mapping constant features to a normalized value of 0.
    2.  **Single-Sample Normalization**: If the normalization is based on a single
        data point, `std` is explicitly set to `1` to prevent undefined behavior.
    3.  **Low-Precision Dtypes**: During the final division, a small epsilon (`1e-16`)
        is added to the denominator. This prevents division by a near-zero `std`,
        which could cause the value to overflow to infinity (`inf`), especially when
        using low-precision dtypes like `torch.float16`.
    4.  **Autograd Compatibility**: All inplace operations (`x[:] = ...`) have been
        replaced with their functional equivalents (`torch.where`, `torch.ones_like`)
        to ensure the computation graph is not corrupted, allowing gradients to be
        computed correctly.

    Args:
        data: The data to normalize. (T, B, H)
        normalize_positions: If > 0, only use the first `normalize_positions`
            positions for normalization.
        return_scaling: If True, return the scaling parameters as well (mean, std).
        std_only: If True, only divide by std.
        clip: If True, clip the data to [-100, 100].
        mean: If given, use this value instead of computing it.
        std: If given, use this value instead of computing it.

    Returns:
        The normalized data tensor, or a tuple containing the data and scaling factors.
    """
    assert (mean is None) == (
        std is None
    ), "Either both or none of mean and std must be given"
    if mean is None:
        if normalize_positions is not None and normalize_positions > 0:
            mean = torch_nanmean(data[:normalize_positions], axis=0)  # type: ignore
            std = torch_nanstd(data[:normalize_positions], axis=0)
        else:
            mean = torch_nanmean(data, axis=0)  # type: ignore
            std = torch_nanstd(data, axis=0)

        # Inplace assignments with functional equivalents to support autograd.
        std = torch.where(std == 0, torch.ones_like(std), std)

        if len(data) == 1 or normalize_positions == 1:
            std = torch.ones_like(std)

        if std_only:
            mean = torch.zeros_like(mean)

    # Add epsilon for numerical stability
    data = (data - mean) / (std + 1e-16)

    if clip:
        data = torch.clip(data, min=-100, max=100)

    if return_scaling:
        return data, (mean, std)
    return data


def select_features(x: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    """Select features from the input tensor based on the selection mask,
    and arrange them contiguously in the last dimension.
    If batch size is bigger than 1, we pad the features with zeros to make the number of features fixed.

    Args:
        x: The input tensor of shape (sequence_length, batch_size, total_features)
        sel: The boolean selection mask indicating which features to keep of shape (batch_size, total_features)

    Returns:
        The tensor with selected features.
        The shape is (sequence_length, batch_size, number_of_selected_features) if batch_size is 1.
        The shape is (sequence_length, batch_size, total_features) if batch_size is greater than 1.
    """
    B, total_features = sel.shape
    sequence_length = x.shape[0]

    # If B == 1, we don't need to append zeros, as the number of features don't need to be fixed.
    if B == 1:
        return x[:, :, sel[0]]

    new_x = torch.zeros(
        (sequence_length, B, total_features),
        device=x.device,
        dtype=x.dtype,
    )

    # For each batch, compute the number of selected features.
    sel_counts = sel.sum(dim=-1)  # shape: (B,)

    for b in range(B):
        s = int(sel_counts[b])
        if s > 0:
            new_x[:, b, :s] = x[:, b, sel[b]]

    return new_x


def remove_outliers(
    X: torch.Tensor,
    n_sigma: float = 4,
    normalize_positions: int = -1,
    lower: None | torch.Tensor = None,
    upper: None | torch.Tensor = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    # Expects T, B, H
    assert (lower is None) == (upper is None), "Either both or none of lower and upper"
    assert len(X.shape) == 3, "X must be T,B,H"
    # for b in range(X.shape[1]):
    # for col in range(X.shape[2]):
    if lower is None:
        data = X if normalize_positions == -1 else X[:normalize_positions]
        data_clean = data[:].clone()
        data_mean, data_std = torch_nanmean(data, axis=0), torch_nanstd(data, axis=0)
        cut_off = data_std * n_sigma
        lower, upper = data_mean - cut_off, data_mean + cut_off

        data_clean[torch.logical_or(data_clean > upper, data_clean < lower)] = np.nan
        data_mean, data_std = (
            torch_nanmean(data_clean, axis=0),
            torch_nanstd(data_clean, axis=0),
        )
        cut_off = data_std * n_sigma
        lower, upper = data_mean - cut_off, data_mean + cut_off

    X = torch.maximum(-torch.log(1 + torch.abs(X)) + lower, X)
    X = torch.minimum(torch.log(1 + torch.abs(X)) + upper, X)
    return X, (lower, upper)


class InputEncoder(nn.Module):
    """Base class for input encoders.

    All input encoders should subclass this class and implement the `forward` method.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, single_eval_pos: int) -> torch.Tensor:
        """Encode the input tensor.

        Args:
            x: The input tensor to encode.
            single_eval_pos: The position to use for single evaluation.

        Returns:
            The encoded tensor.
        """
        raise NotImplementedError


class SequentialEncoder(nn.Sequential, InputEncoder):
    """An encoder that applies a sequence of encoder steps.

    SequentialEncoder allows building an encoder from a sequence of EncoderSteps.
    The input is passed through each step in the provided order.
    """

    def __init__(self, *args: SeqEncStep, output_key: str = "output", **kwargs: Any):
        """Initialize the SequentialEncoder.

        Args:
            *args: A list of SeqEncStep instances to apply in order.
            output_key:
                The key to use for the output of the encoder in the state dict.
                Defaults to "output", i.e. `state["output"]` will be returned.
            **kwargs: Additional keyword arguments passed to `nn.Sequential`.
        """
        super().__init__(*args, **kwargs)
        self.output_key = output_key

    def forward(self, input: dict, **kwargs: Any) -> torch.Tensor:
        """Apply the sequence of encoder steps to the input.

        Args:
            input:
                The input state dictionary.
                If the input is not a dict and the first layer expects one input key,
                the input tensor is mapped to the key expected by the first layer.
            **kwargs: Additional keyword arguments passed to each encoder step.

        Returns:
            The output of the final encoder step.
        """
        # If the input is not a dict and the first layer expects one input, mapping the
        #   input to the first input key must be correct
        if not isinstance(input, dict) and len(self[0].in_keys) == 1:
            input = {self[0].in_keys[0]: input}

        for module in self:
            input = module(input, **kwargs)

        return input[self.output_key] if self.output_key is not None else input


class SeqEncStep(nn.Module):
    """Abstract base class for sequential encoder steps.

    SeqEncStep is a wrapper around a module that defines the expected input keys
    and the produced output keys. The outputs are assigned to the output keys
    in the order specified by `out_keys`.

    Subclasses should either implement `_forward` or `_fit` and `_transform`.
    Subclasses that transform `x` should always use `_fit` and `_transform`,
    creating any state that depends on the train set in `_fit` and using it in `_transform`.
    This allows fitting on data first and doing inference later without refitting.
    Subclasses that work with `y` can alternatively use `_forward` instead.
    """

    def __init__(
        self,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main",),
    ):
        """Initialize the SeqEncStep.

        Args:
            in_keys: The keys of the input tensors.
            out_keys: The keys to assign the output tensors to.
        """
        super().__init__()
        self.in_keys = in_keys
        self.out_keys = out_keys

    # Either implement _forward:

    def _forward(self, *x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        """Forward pass of the encoder step.

        Implement this if not implementing _fit and _transform.

        Args:
            *x: The input tensors. A single tensor or a tuple of tensors.
            **kwargs: Additional keyword arguments passed to the encoder step.

        Returns:
            The output tensor or a tuple of output tensors.
        """
        raise NotImplementedError()

    # Or implement _fit and _transform:

    def _fit(
        self,
        *x: torch.Tensor,
        single_eval_pos: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Fit the encoder step on the training set.

        Args:
            *x: The input tensors. A single tensor or a tuple of tensors.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments passed to the encoder step.
        """
        raise NotImplementedError

    def _transform(
        self,
        *x: torch.Tensor,
        single_eval_pos: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor]:
        """Transform the data using the fitted encoder step.

        Args:
            *x: The input tensors. A single tensor or a tuple of tensors.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments passed to the encoder step.

        Returns:
            The transformed output tensor or a tuple of output tensors.
        """
        raise NotImplementedError

    def forward(
        self,
        state: dict,
        cache_trainset_representation: bool = False,
        **kwargs: Any,
    ) -> dict:
        """Perform the forward pass of the encoder step.

        Args:
            state: The input state dictionary containing the input tensors.
            cache_trainset_representation:
                Whether to cache the training set representation. Only supported for
                _fit and _transform (not _forward).
            **kwargs: Additional keyword arguments passed to the encoder step.

        Returns:
            The updated state dictionary with the output tensors assigned to the output keys.
        """
        args = [state[in_key] for in_key in self.in_keys]
        if hasattr(self, "_fit"):
            if kwargs["single_eval_pos"] or not cache_trainset_representation:
                self._fit(*args, **kwargs)
            out = self._transform(*args, **kwargs)
        else:
            assert not cache_trainset_representation
            out = self._forward(*args, **kwargs)
            # TODO: I think nothing is using _forward now

        assert isinstance(
            out,
            tuple,
        ), f"out is not a tuple: {out}, type: {type(out)}, class: {self.__class__.__name__}"
        assert len(out) == len(self.out_keys)
        state.update({out_key: out[i] for i, out_key in enumerate(self.out_keys)})
        return state


class LinearInputEncoderStep(SeqEncStep):
    """A simple linear input encoder step."""

    def __init__(
        self,
        *,
        num_features: int,
        emsize: int,
        replace_nan_by_zero: bool = False,
        bias: bool = True,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("output",),
    ):
        """Initialize the LinearInputEncoderStep.

        Args:
            num_features: The number of input features.
            emsize: The embedding size, i.e. the number of output features.
            replace_nan_by_zero: Whether to replace NaN values in the input by zero. Defaults to False.
            bias: Whether to use a bias term in the linear layer. Defaults to True.
            in_keys: The keys of the input tensors. Defaults to ("main",).
            out_keys: The keys to assign the output tensors to. Defaults to ("output",).
        """
        super().__init__(in_keys, out_keys)
        self.layer = nn.Linear(num_features, emsize, bias=bias)
        self.replace_nan_by_zero = replace_nan_by_zero

    def _fit(self, *x: torch.Tensor, **kwargs: Any):
        """Fit the encoder step. Does nothing for LinearInputEncoderStep."""

    def _transform(self, *x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        """Apply the linear transformation to the input.

        Args:
            *x: The input tensors to concatenate and transform.
            **kwargs: Unused keyword arguments.

        Returns:
            A tuple containing the transformed tensor.
        """
        x = torch.cat(x, dim=-1)
        if self.replace_nan_by_zero:
            x = torch.nan_to_num(x, nan=0.0)  # type: ignore

        # Ensure input tensor dtype matches the layer's weight dtype
        # Since this layer gets input from the outside we verify the dtype
        x = x.to(self.layer.weight.dtype)

        return (self.layer(x),)


class NanHandlingEncoderStep(SeqEncStep):
    """Encoder step to handle NaN and infinite values in the input."""

    nan_indicator = -2.0
    inf_indicator = 2.0
    neg_inf_indicator = 4.0

    def __init__(
        self,
        keep_nans: bool = True,
        in_keys: tuple[str, ...] = ("main",),
        out_keys: tuple[str, ...] = ("main", "nan_indicators"),
    ):
        """Initialize the NanHandlingEncoderStep.

        Args:
            keep_nans: Whether to keep NaN values as separate indicators. Defaults to True.
            in_keys: The keys of the input tensors. Must be a single key.
            out_keys: The keys to assign the output tensors to.
        """
        assert len(in_keys) == 1, "NanHandlingEncoderStep expects a single input key"
        super().__init__(in_keys, out_keys)
        self.keep_nans = keep_nans
        self.register_buffer("feature_means_", torch.tensor([]), persistent=False)

    def _fit(self, x: torch.Tensor, single_eval_pos: int, **kwargs: Any) -> None:
        """Compute the feature means on the training set for replacing NaNs.

        Args:
            x: The input tensor.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments (unused).
        """
        self.feature_means_ = torch_nanmean(
            x[:single_eval_pos],
            axis=0,
            include_inf=True,
        )

    def _transform(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace NaN and infinite values in the input tensor.

        Args:
            x: The input tensor.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the transformed tensor and optionally the NaN indicators.
        """
        nans_indicator = None
        if self.keep_nans:
            # TODO: There is a bug here: The values arriving here are already mapped to nan if they were inf before
            nans_indicator = (
                torch.isnan(x) * NanHandlingEncoderStep.nan_indicator
                + torch.logical_and(torch.isinf(x), torch.sign(x) == 1)
                * NanHandlingEncoderStep.inf_indicator
                + torch.logical_and(torch.isinf(x), torch.sign(x) == -1)
                * NanHandlingEncoderStep.neg_inf_indicator
            ).to(x.dtype)

        nan_mask = torch.logical_or(torch.isnan(x), torch.isinf(x))
        # replace nans with the mean of the corresponding feature
        x = x.clone()  # clone to avoid inplace operations
        x[nan_mask] = self.feature_means_.unsqueeze(0).expand_as(x)[nan_mask]
        return x, nans_indicator


class RemoveEmptyFeaturesEncoderStep(SeqEncStep):
    """Encoder step to remove empty (constant) features.
    Was changed to NOT DO ANYTHING, the removal of empty features now
    done elsewhere, but the saved model still needs this encoder step.
    TODO: REMOVE.
    """

    def __init__(self, **kwargs: Any):
        """Initialize the RemoveEmptyFeaturesEncoderStep.

        Args:
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.sel = None

    def _fit(self, x: torch.Tensor, **kwargs: Any) -> None:
        """Compute the feature selection mask on the training set.

        Args:
            x: The input tensor.
            **kwargs: Additional keyword arguments (unused).
        """
        self.sel = (x[1:] == x[0]).sum(0) != (x.shape[0] - 1)

    def _transform(self, x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        """Remove empty features from the input tensor.

        Args:
            x: The input tensor.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the transformed tensor with empty features removed.
        """
        return (select_features(x, self.sel),)


class RemoveDuplicateFeaturesEncoderStep(SeqEncStep):
    """Encoder step to remove duplicate features."""

    def __init__(self, normalize_on_train_only: bool = True, **kwargs: Any):
        """Initialize the RemoveDuplicateFeaturesEncoderStep.

        Args:
            normalize_on_train_only: Whether to normalize only on the training set.
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.normalize_on_train_only = normalize_on_train_only

    def _fit(self, x: torch.Tensor, single_eval_pos: int, **kwargs: Any) -> None:
        """Currently does nothing. Fit functionality not implemented."""

    def _transform(
        self,
        x: torch.Tensor,
        single_eval_pos: int,
        **kwargs: Any,
    ) -> tuple[torch.Tensor]:
        """Remove duplicate features from the input tensor.

        Args:
            x: The input tensor.
            single_eval_pos: The position to use for single evaluation.
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the input tensor (removal not implemented).
        """
        # TODO: This uses a lot of memory, as it computes the covariance matrix for each batch
        #   This could be done more efficiently, models go OOM with this
        return (x,)
        normalize_position = single_eval_pos if self.normalize_on_train_only else -1

        x_norm = normalize_data(x[:, :normalize_position])
        sel = torch.zeros(x.shape[1], x.shape[2], dtype=torch.bool, device=x.device)
        for B in range(x_norm.shape[1]):
            cov_mat = (torch.cov(x_norm[:, B].transpose(1, 0)) > 0.999).float()
            cov_mat_sum_below_trace = torch.triu(cov_mat).sum(dim=0)
            sel[B] = cov_mat_sum_below_trace == 1.0

        new_x = select_features(x, sel)

        return (new_x,)


class VariableNumFeaturesEncoderStep(SeqEncStep):
    """Encoder step to handle variable number of features.

    Transforms the input to a fixed number of features by appending zeros.
    Also normalizes the input by the number of used features to keep the variance
    of the input constant, even when zeros are appended.
    """

    def __init__(
        self,
        num_features: int,
        normalize_by_used_features: bool = True,
        normalize_by_sqrt: bool = True,
        **kwargs: Any,
    ):
        """Initialize the VariableNumFeaturesEncoderStep.

        Args:
            num_features: The number of features to transform the input to.
            normalize_by_used_features: Whether to normalize by the number of used features.
            normalize_by_sqrt: Legacy option to normalize by sqrt instead of the number of used features.
            **kwargs: Keyword arguments passed to the parent SeqEncStep.
        """
        super().__init__(**kwargs)
        self.normalize_by_used_features = normalize_by_used_features
        self.num_features = num_features
        self.normalize_by_sqrt = normalize_by_sqrt
        self.number_of_used_features_ = None

    def _fit(self, x: torch.Tensor, **kwargs: Any) -> None:
        """Compute the number of used features on the training set.

        Args:
            x: The input tensor.
            **kwargs: Additional keyword arguments (unused).
        """
        sel = (x[1:] == x[0]).sum(0) != (x.shape[0] - 1)
        self.number_of_used_features_ = torch.clip(
            sel.sum(-1).unsqueeze(-1),
            min=1,
        ).cpu()

    def _transform(self, x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor]:
        """Transform the input tensor to have a fixed number of features.

        Args:
            x: The input tensor of shape (seq_len, batch_size, num_features_old).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            A tuple containing the transformed tensor of shape (seq_len, batch_size, num_features).
        """
        if x.shape[2] == 0:
            return (
                torch.zeros(
                    x.shape[0],
                    x.shape[1],
                    self.num_features,
                    device=x.device,
                    dtype=x.dtype,
                ),
            )
        if self.normalize_by_used_features:
            if self.normalize_by_sqrt:
                # Verified that this gives indeed unit variance with appended zeros
                x = x * torch.sqrt(
                    self.num_features / self.number_of_used_features_.to(x.device),
                )
            else:
                x = x * (self.num_features / self.number_of_used_features_.to(x.device))

        zeros_appended = torch.zeros(
            *x.shape[:-1],
            self.num_features - x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat([x, zeros_appended], -1)
        return (x,)


class InputNormalizationEncoderStep(SeqEncStep):
    """Encoder step to normalize the input in different ways.

    Can be used to normalize the input to a ranking, remove outliers,
    or normalize the input to have unit variance.
    """

    def __init__(
        self,
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
            normalize_on_train_only: Whether to compute normalization only on the training set.
            normalize_to_ranking: Whether to normalize the input to a ranking.
            normalize_x: Whether to normalize the input to have unit variance.
            remove_outliers: Whether to remove outliers from the input.
            remove_outliers_sigma: The number of standard deviations to use for outlier removal.
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
        self.lower_for_outlier_removal = None
        self.upper_for_outlier_removal = None
        self.mean_for_normalization = None
        self.std_for_normalization = None

    def reset_seed(self) -> None:
        """Reset the random seed."""

    def _fit(self, x: torch.Tensor, single_eval_pos: int, **kwargs: Any) -> None:
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
        **kwargs: Any,
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
                "Not implemented currently as it was not used in a long time and hard to move out the state.",
            )
            x = to_ranking_low_mem(x)

        if self.remove_outliers:
            assert (
                self.remove_outliers_sigma > 1.0
            ), "remove_outliers_sigma must be > 1.0"

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
    ):
        """Fit the encoder step. Does nothing for FrequencyFeatureEncoderStep."""

    def _transform(
        self,
        x: torch.Tensor,
        single_eval_pos: int | None = None,
        categorical_inds: list[int] | None = None,
    ):
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


class CategoricalInputEncoderPerFeatureEncoderStep(SeqEncStep):
    """Expects input of size 1."""

    def __init__(
        self,
        num_features,
        emsize,
        base_encoder,
        num_embs: int = 1_000,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        assert num_features == 1
        self.num_features = num_features
        self.emsize = emsize
        self.num_embs = num_embs
        self.embedding = nn.Embedding(num_embs, emsize)
        self.base_encoder = base_encoder

    def _fit(self, x, single_eval_pos: int, categorical_inds: list[int]):
        pass

    def _transform(
        self,
        x,
        single_eval_pos: int,
        categorical_inds: list[int],
    ):
        if categorical_inds is None:
            is_categorical = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
        else:
            assert all(ci in ([0], []) for ci in categorical_inds), categorical_inds
            is_categorical = torch.tensor(
                [ci == [0] for ci in categorical_inds],
                device=x.device,
            )

        if is_categorical.any():
            lx = x[:, is_categorical]
            nan_mask = torch.isnan(lx) | torch.isinf(lx)
            lx = lx.long().clamp(0, self.num_embs - 2)
            lx[nan_mask] = self.num_embs - 1
            categorical_embs = self.embedding(lx.squeeze(-1))
        else:
            categorical_embs = torch.zeros(x.shape[0], 0, x.shape[2], device=x.device)

        if (~is_categorical).any():
            lx = x[:, ~is_categorical]
            continuous_embs = self.base_encoder(lx, single_eval_pos=single_eval_pos)[0]
        else:
            continuous_embs = torch.zeros(x.shape[0], 0, x.shape[2], device=x.device)

        # return (torch.cat((continuous_embs, categorical_embs), dim=1),)
        # above is wrong as we need to preserve order in the batch dimension
        embs = torch.zeros(
            x.shape[0],
            x.shape[1],
            self.emsize,
            device=x.device,
            dtype=torch.float,
        )
        embs[:, is_categorical] = categorical_embs.float()
        embs[:, ~is_categorical] = continuous_embs.float()
        return (embs,)


class StyleEncoder(nn.Module):
    def __init__(self, num_hyperparameters, em_size):
        super().__init__()
        self.em_size = em_size
        self.embedding = nn.Linear(num_hyperparameters, self.em_size)

    def forward(self, hyperparameters):  # B x num_hps
        return self.embedding(hyperparameters)


def get_linear_encoder_generator(in_keys):
    def get_linear_encoder(num_features, emsize):
        return SequentialEncoder(
            LinearInputEncoderStep(
                num_features,
                emsize,
                in_keys=in_keys,
                out_keys=["output"],
            ),
            output_key="output",
        )

    return get_linear_encoder


##### TARGET ENCODERS #####


class MulticlassClassificationTargetEncoder(SeqEncStep):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.unique_ys_ = None

    def _fit(self, y: torch.Tensor, single_eval_pos: int, **kwargs: Any):
        assert len(y.shape) == 3 and (y.shape[-1] == 1), "y must be of shape (T, B, 1)"
        self.unique_ys_ = [
            torch.unique(y[:single_eval_pos, b_i]) for b_i in range(y.shape[1])
        ]

    @staticmethod
    def flatten_targets(y: torch.Tensor, unique_ys: torch.Tensor | None = None):
        if unique_ys is None:
            unique_ys = torch.unique(y)
        return (y.unsqueeze(-1) > unique_ys).sum(axis=-1)

    def _transform(self, y: torch.Tensor, single_eval_pos: int | None = None):
        assert len(y.shape) == 3 and (y.shape[-1] == 1), "y must be of shape (T, B, 1)"
        assert not (
            y.isnan().any() and self.training
        ), "NaNs are not allowed in the target at this point during training (set to model.eval() if not in training)"
        y_new = y.clone()
        for B in range(y.shape[1]):
            y_new[:, B, :] = self.flatten_targets(y[:, B, :], self.unique_ys_[B])
        return (y_new,)
