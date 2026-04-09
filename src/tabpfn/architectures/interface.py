"""Defines the interface for modules containing architectures."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Literal, Protocol, overload
from typing_extensions import override

from pydantic.dataclasses import dataclass
from torch import Tensor, nn


@dataclass
class ArchitectureConfig:
    """Base configuration class that each architecture config should inherit from.

    Contains config keys common to all the architectures.
    """

    max_num_classes: int = -1
    """Maximum number of classes the model should support.

    Must be set to a value greater than 0 to support classification."""
    num_buckets: int = -1
    """In regression models: the number of buckets in the output bar distribution.

    Must be set to a value greater than 0 to support regression.
    """

    def get_unused_config(self, unparsed_config: dict[str, Any]) -> dict[str, Any]:
        """Returns items in the given config that were not parsed by this config.

        This emulates Pydantic's extra="allow" and __pydantic_extra__ feature, which
        unfortunately isn't supported for dataclasses.
        """
        return _get_unused_items(full_config=unparsed_config, used_config=asdict(self))


def _get_unused_items(
    full_config: dict[str, Any], used_config: dict[str, Any]
) -> dict[str, Any]:
    unused = {}
    for k, v in full_config.items():
        if k not in used_config:
            unused[k] = v
        elif isinstance(v, dict):
            subconfig_unused = _get_unused_items(v, used_config[k])
            if len(subconfig_unused) > 0:
                unused[k] = subconfig_unused
    return unused


@dataclasses.dataclass(frozen=True)
class PerformanceOptions:
    """Options controlling performance/memory trade-offs in the forward pass.

    Pass an instance of this class as the `performance_options` argument to
    `Architecture.forward` to tune memory usage and compute behaviour for a
    single call without changing model weights or configuration.

    These are purely optional performance tweaks.  Each architecture may
    choose to support, partially support, or silently ignore any individual
    option — they will never raise an error if an unsupported option is passed.
    """

    save_peak_memory_factor: int | None = None
    """Chunk factor for within-layer memory saving (attention, MLP, layer norm).

    When set, the attention computation, MLP, and layer-norm steps are split into
    `save_peak_memory_factor` chunks and executed sequentially, avoiding the need
    to materialise the full intermediate tensors.  `None` disables chunking
    (default).  Higher values reduce peak GPU memory, and can also increase
    throughput by reducing memory pressure and the number of CPU<->GPU
    synchronisation points required for memory allocations."""

    force_recompute_layer: bool = False
    """Enable activation checkpointing (gradient recomputation) for all layers.

    When ``True``, intermediate activations are not stored during the forward pass;
    instead they are recomputed from scratch during the backward pass.  This trades
    compute for memory and is useful when training with very large context sizes.
    Has no effect during inference (``torch.no_grad`` / ``torch.inference_mode``).
    """

    use_chunkwise_inference: bool = False
    """Use the chunked inference path that avoids materialising `(B, Ri, C, E)`.

    When `True`, the decoder iterates over test rows and feature groups in small
    chunks so that the full `(batch, rows, cols, embed)` tensor is never fully
    resident in memory at once.
    """


class ArchitectureModule(Protocol):
    """Interface that modules containing model architectures should implement."""

    def parse_config(
        self, config: dict[str, Any]
    ) -> tuple[ArchitectureConfig, dict[str, Any]]:
        """Parses the given config dict into ArchitectureConfig or a subclass.

        This config will then be passed to get_architecture(), in order to construct the
        architecture object. This architecture should subclass ArchitectureConfig as
        necessary, to add its own keys.

        Unrecognised keys should be ignored during parsing, and returned in the `unused
        config items` dict.

        Args:
            config: Config dict to parse. This function should use Pydantic to
                verify that it matches the expected schema.

        Returns: a tuple (the parsed config, dict containing unused config items)

        Raises:
            pydantic.ValidationError: one or more of the values have the wrong type
        """
        ...

    def get_architecture(
        self,
        config: ArchitectureConfig,
        *,
        cache_trainset_representation: bool,
    ) -> Architecture:
        """Construct a new instance of the model based on the given config.

        Args:
            config: The config returned by parse_config(). This method should use a
                runtime isinstance() check to downcast the config to this architecture's
                specific config class.
            cache_trainset_representation: If True, the model should be configured to
                cache the training data during inference to improve speed.

        Returns: the constructed architecture
        """
        ...


class Architecture(nn.Module, ABC):
    """The interface that all architectures must implement.

    Architectures are PyTorch modules, which is then wrapped by e.g.
    TabPFNClassifier or TabPFNRegressor to form the complete model.
    """

    @overload
    @abstractmethod
    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        y: Tensor | dict[str, Tensor] | None,
        *,
        only_return_standard_out: Literal[True] = True,
        categorical_inds: list[list[int]] | None = None,
        performance_options: PerformanceOptions | None = None,
        task_type: str | None = None,
    ) -> Tensor: ...

    @overload
    @abstractmethod
    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        y: Tensor | dict[str, Tensor] | None,
        *,
        only_return_standard_out: Literal[False],
        categorical_inds: list[list[int]] | None = None,
        performance_options: PerformanceOptions | None = None,
        task_type: str | None = None,
    ) -> dict[str, Tensor]: ...

    @abstractmethod
    @override
    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        y: Tensor | dict[str, Tensor] | None,
        *,
        only_return_standard_out: bool = True,
        categorical_inds: list[list[int]] | None = None,
        performance_options: PerformanceOptions | None = None,
        task_type: str | None = None,
    ) -> Tensor | dict[str, Tensor]:
        """Perform a forward pass.

        Args:
            x: The input data. Either:
                - A Tensor with shape
                  `[(train+test) rows, batch size, num input features]`.
                - A dictionary containing at least `{"main": x}`, where `x` is the
                  Tensor above. The dictionary may also contain additional keys, which
                  are relevant for particular encoders.

            y: The target data. Either:
                - A Tensor with shape `(train rows)`, `(train_rows, batch_size)`, or
                  shape `(train_rows, batch_size, 1)`.
                - A dictionary containing at least `{"main": y}`, where `y` is the
                  Tensor above. The dictionary may also contain additional keys, which
                  are relevant for particular encoders.
                - `None`, if there are no training rows, as when making predictions
                  using the KV cache.

            only_return_standard_out: Configures the return value, see "Returns" section
                below.

            categorical_inds: The indices of categorical features.

            performance_options: Performance and memory options for this forward pass.
                If None, uses defaults (no memory saving, no recomputation).
            task_type: The type of task, typically "classification" or "regression".

        Returns:
            If `only_return_standard_out`, then a Tensor of shape
            `(test rows, batch size, num classes)`, which is the output of the
            standard decoder.
            Otherwise, a dictionary containing the output of each decoder, and also:
                - "train_embeddings": The output of the encoder on the training data.
                - "test_embeddings": The output of the encoder on the test data.
            Particular models may also return additional information.
        """
        ...

    def get_default_performance_options(self) -> PerformanceOptions:
        """Return the default :class:`PerformanceOptions` for this architecture.

        Subclasses may override this to change the defaults, e.g. to enable
        memory-efficient inference by default for large models.
        """
        return PerformanceOptions(
            save_peak_memory_factor=None,
            force_recompute_layer=False,
            use_chunkwise_inference=False,
        )
