#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import os
import warnings
from collections.abc import Callable
from types import MethodType
from typing import Any, Literal

import numpy as np
import torch

# TODO(eddiebergman): Make this an option
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
SAVE_PEAK_MEM_FACTOR = 8

# TODO(eddiebergman): pulled from `def _estimate_model_usage()`
CONSTANT_MEMORY_OVERHEAD = 100_000_000
MEMORY_FACTOR_SAVE_PEAK_MEM_ACTIVE = 2.5
DEFAULT_CPU_MEMORY_GB_IF_NOT_CUDA = 8

# TODO(eddiebergman): pulled from `def _estimate_model_usage()`
# Had it's own todo of "check if correct"
NUM_SAMPLES_FACTOR = 4
NUM_SAMPLES_PLUS_FEATURES = 6.5
CELLS_FACTOR = 0.25
CELLS_SQUARED_FACTOR = 1.3e-7

TO_BYTES_CONVERSION = {"b": 1, "mb": 1e6, "gb": 1e9}


def support_save_peak_mem_factor(method: MethodType) -> Callable:
    """Can be applied to a method acting on a tensor 'x' whose first dimension is a
    flat batch dimension
    (i.e. the operation is trivially parallel over the first dimension).

    For additional tensor arguments, it is assumed that the first dimension is again
    the batch dimension, and that non-tensor arguments can be passed as-is
    to splits when parallelizing over the batch dimension.

    The decorator adds options 'add_input' to add the principal input 'x' to the
    result of the method and 'allow_inplace'.
    By setting 'allow_inplace', the caller indicates that 'x'
    is not used after the call and its buffer can be reused for the output.

    Setting 'allow_inplace' does not ensure that the operation will be inplace,
    and the return value should be used for clarity and simplicity.

    Moreover, it adds an optional int parameter 'save_peak_mem_factor' that is
    only supported in combination with 'allow_inplace' during inference and subdivides
    the operation into the specified number of chunks to reduce peak memory consumption.
    """

    def method_(
        self: torch.nn.Module,
        x: torch.Tensor,
        *args: torch.Tensor,
        add_input: bool = False,
        allow_inplace: bool = False,
        save_peak_mem_factor: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        assert isinstance(self, torch.nn.Module)
        assert (
            save_peak_mem_factor is None or allow_inplace
        ), "The parameter save_peak_mem_factor only supported with 'allow_inplace' set."
        assert isinstance(x, torch.Tensor)

        tensor_inputs = list(tuple(self.parameters()) + tuple(args))

        assert (
            save_peak_mem_factor is None
            or not any(t.requires_grad for t in tensor_inputs)
            or not torch.is_grad_enabled()
        ), "The parameter save_peak_mem_factor is only supported during inference."

        if save_peak_mem_factor is not None:
            assert isinstance(save_peak_mem_factor, int)
            assert save_peak_mem_factor > 1
            split_size = (x.size(0) + save_peak_mem_factor - 1) // save_peak_mem_factor

            split_args = zip(
                *[
                    torch.split(arg, split_size)
                    if isinstance(arg, torch.Tensor)
                    else [arg] * save_peak_mem_factor
                    for arg in (x, *args)
                ],
            )

            for x_, *args_ in split_args:
                if add_input:
                    x_[:] += method(self, x_, *args_, **kwargs)
                else:
                    x_[:] = method(self, x_, *args_, **kwargs)
            return x

        if add_input:
            return x + method(self, x, *args, **kwargs)

        return method(self, x, *args, **kwargs)

    return method_


class MemoryUsageEstimator:
    SAVE_PEAK_MEM_FACTOR = 8

    @classmethod
    def convert_units(
        cls,
        value: float,
        from_unit: Literal["b", "mb", "gb"],
        to_unit: Literal["b", "mb", "gb"],
    ) -> float:
        """Convert a value from one unit to another."""
        if from_unit not in TO_BYTES_CONVERSION:
            raise ValueError(
                f"Invalid unit {from_unit}. Must be one of 'b', 'mb', or 'gb'.",
            )
        if to_unit not in TO_BYTES_CONVERSION:
            raise ValueError(
                f"Invalid unit {to_unit}. Must be one of 'b', 'mb', or 'gb'.",
            )

        return (value * TO_BYTES_CONVERSION[from_unit]) / TO_BYTES_CONVERSION[to_unit]

    @classmethod
    def convert_bytes_to_unit(
        cls,
        value: float,
        unit: Literal["b", "mb", "gb"],
    ) -> float:
        """Convenience method to convert bytes to a different unit.

        Args:
            value: The number of bytes.
            unit: The unit to convert to.

        Returns:
            The number of bytes in the new unit.
        """
        return cls.convert_units(value, "b", unit)

    @classmethod
    def estimate_memory_of_one_batch(
        cls,
        X: torch.Tensor,
        model: torch.nn.Module,
        *,
        cache_kv: bool,
        dtype_byte_size: int,
        unit: Literal["b", "mb", "gb"] = "gb",
        n_train_samples: int | None = None,
    ) -> float:
        """Estimate the memory usage of a single batch.

        The calculation is done based on the assumption that save_peak_mem_factor
        is not used (since this estimation is used to determine whether to use it).

        Args:
            X: The input tensor.
            model: The model to estimate the memory usage of.
            cache_kv: Whether key and value tensors are cached.
            dtype_byte_size: The size of the data type in bytes.
            unit: The unit to convert the memory usage to.
            n_train_samples: The number of training samples (only for cache_kv mode)

        Returns:
            The estimated memory usage of a single batch.
        """
        assert len(X.shape) in (2, 3), "X must be a 2D or 3D tensor"

        if cache_kv:
            assert isinstance(
                n_train_samples,
                int,
            ), "n_train_samples must be provided when cache_kv is True"

        if unit not in TO_BYTES_CONVERSION:
            raise ValueError(f"Invalid unit {unit}. Must be one of 'b', 'mb', or 'gb'.")

        embedding_size = model.ninp
        features_per_group = model.features_per_group

        n_layers = None
        # Assumes the model has only encoder blocks
        if (
            hasattr(model, "transformer_encoder")
            and model.transformer_encoder is not None
        ):
            n_layers = len(model.transformer_encoder.layers)

        # Guarding against future changes in the transformer model
        # Ideally, there should be an API exposed in the model to get the
        # number of layers
        if n_layers is None:
            n_layers = 12
            warnings.warn(
                "Could not estimate number of encoder/decoder layers in the "
                "transformer model, defaulting to 12.",
                stacklevel=2,
            )

        n_samples, n_features = X.shape[-2], X.shape[-1]
        n_batches = X.shape[0] if len(X.shape) == 3 else 1

        n_feature_groups = int(np.ceil(n_features / features_per_group)) + 1

        model_mem = sum(p.numel() for p in model.parameters()) * dtype_byte_size
        X_mem = n_samples * n_feature_groups * dtype_byte_size
        activation_mem = (
            n_samples
            * n_feature_groups
            * embedding_size
            * n_layers
            * dtype_byte_size
            * n_batches
        )

        total_mem_bytes = model_mem + X_mem + activation_mem

        if cache_kv:
            cached_mem = (
                n_train_samples  # type: ignore
                * n_feature_groups
                * embedding_size
                * 2  # key and value
                * n_layers
                * dtype_byte_size
            )
            total_mem_bytes += cached_mem

        return cls.convert_bytes_to_unit(total_mem_bytes, unit)

    @classmethod
    def get_max_free_memory(
        cls,
        device: torch.device,
        *,
        unit: Literal["b", "mb", "gb"] = "gb",
        default_gb_cpu_if_failed_to_calculate: float,
    ) -> float:
        """How much memory to use at most in GB, the memory usage will be calculated
        based on an estimation of the systems free memory.

        For CUDA will use the free memory of the GPU. For CPU will default to 32 GB.

        Returns:
        -------
        The maximum memory usage in GB.
        """
        # TODO(Arjun): Make it accept a value for GPU specified by the user

        # TODO: Get System Stats and adapt to free memory for default case

        if device.type.startswith("cpu"):
            try:
                free_memory = (
                    os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
                )
            except AttributeError:
                from tabpfn.utils import get_total_memory_windows

                if os.name == "nt":
                    free_memory = get_total_memory_windows()
                else:
                    warnings.warn(
                        "Could not get system memory, defaulting to"
                        f" {default_gb_cpu_if_failed_to_calculate} GB",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    free_memory = cls.convert_units(
                        default_gb_cpu_if_failed_to_calculate,
                        "gb",
                        "b",
                    )

            except ValueError:
                warnings.warn(
                    "Could not get system memory, defaulting to"
                    f" {default_gb_cpu_if_failed_to_calculate} GB",
                    RuntimeWarning,
                    stacklevel=2,
                )
                free_memory = cls.convert_units(
                    default_gb_cpu_if_failed_to_calculate,
                    "gb",
                    "b",
                )

        elif device.type.startswith("cuda"):
            t = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            free_memory = t - a  # free inside reserved
        elif device.type.startswith("mps"):
            # It seems like we would want to use the following functions:
            #    * torch.mps.recommended_max_memory
            #    * torch.mps.current_allocated_memory
            # Not entirely sure of the behavior of the first function
            # so this might not work as intended.
            raise NotImplementedError(
                "Memory estimation for MPS devices is not currently supported."
                " If you have experience with getting memory information for MPS"
                " devices, we would gladly appreciate a contribution!"
                "",
            )
        else:
            raise ValueError(f"Unknown device {device}")

        return cls.convert_bytes_to_unit(free_memory, unit)

    @classmethod
    def estimate_memory_remainder_after_batch(
        cls,
        X: torch.Tensor,
        model: torch.nn.Module,
        *,
        cache_kv: bool,
        device: torch.device,
        dtype_byte_size: int,
        safety_factor: float,
        n_train_samples: int | None = None,
        max_free_mem: float | int | None = None,
    ) -> float:
        """Whether to save peak memory or not.

        Args:
            X: The input tensor.
            model: The model to estimate the memory usage of.
            cache_kv: Whether key and value tensors are cached.
            device: The device to use.
            dtype_byte_size: The size of the data type in bytes.
            safety_factor: The safety factor to apply.
            n_train_samples: The number of training samples (only for cache_kv mode)
            max_free_mem: The amount of free memory available.

        Returns:
            The amount of free memory available after a batch is computed.
        """
        if max_free_mem is None:
            max_free_mem = cls.get_max_free_memory(
                device,
                unit="gb",
                default_gb_cpu_if_failed_to_calculate=DEFAULT_CPU_MEMORY_GB_IF_NOT_CUDA,
            )

        mem_per_batch = cls.estimate_memory_of_one_batch(
            X,
            model,
            cache_kv=cache_kv,
            dtype_byte_size=dtype_byte_size,
            unit="gb",
            n_train_samples=n_train_samples,
        )

        return max_free_mem - (mem_per_batch * safety_factor)

    @classmethod
    def reset_peak_memory_if_required(
        cls,
        save_peak_mem: bool | Literal["auto"] | float | int,
        model: torch.nn.Module,
        X: torch.Tensor,
        *,
        cache_kv: bool,
        device: torch.device,
        dtype_byte_size: int,
        safety_factor: float = 5.0,
        n_train_samples: int | None = None,
    ) -> None:
        """Reset the peak memory if required.

        Args:
            save_peak_mem (bool | "auto" | float | int): If bool, specifies whether to
                save peak memory or not.
                If "auto", the amount of free memory is estimated and the option is
                enabled or disabled based on the estimated usage.
                If float or int, it is considered as the amount of memory available
                (in GB) explicitly specified by the user. In this case, this value is
                used to estimate whether or not to save peak memory.
            model (torch.nn.Module): The model to reset the peak memory of.
            X (torch.Tensor): The input tensor.
            cache_kv (bool): Whether key and value tensors are cached.
            device (torch.device): The device to use.
            dtype_byte_size (int): The size of the data type in bytes.
            safety_factor (float): The safety factor to apply.
            n_train_samples (int): The number of training samples (to be used
                only for cache_kv mode)
        """
        save_peak_mem_is_num = isinstance(
            save_peak_mem,
            (float, int),
        ) and not isinstance(save_peak_mem, bool)
        if save_peak_mem == "auto" or save_peak_mem_is_num:
            memory_available_after_batch = cls.estimate_memory_remainder_after_batch(
                X,
                model,
                cache_kv=cache_kv,
                device=device,
                dtype_byte_size=dtype_byte_size,
                safety_factor=safety_factor,
                n_train_samples=n_train_samples,
                max_free_mem=save_peak_mem
                if isinstance(save_peak_mem, (float, int))
                else None,
            )
            save_peak_mem = memory_available_after_batch < 0

        if save_peak_mem:
            model.reset_save_peak_mem_factor(cls.SAVE_PEAK_MEM_FACTOR)
        else:
            model.reset_save_peak_mem_factor(None)
