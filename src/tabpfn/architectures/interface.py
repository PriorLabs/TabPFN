"""Defines the interface for modules containing architectures."""

from __future__ import annotations

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

    max_num_classes: int
    num_buckets: int
    """In regression models: the number of buckets in the output bar distribution.

    In classification models, does nothing.
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
        n_out: int,
        cache_trainset_representation: bool,
    ) -> Architecture:
        """Construct a new instance of the model based on the given config.

        Args:
            config: The config returned by parse_config(). This method should use a
                runtime isinstance() check to downcast the config to this architecture's
                specific config class.
            n_out: The number of output classes that the model should predict.
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
            only_return_standard_out: Whether to only return the standard output.
            categorical_inds: The indices of categorical features.

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
