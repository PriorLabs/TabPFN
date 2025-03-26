#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import dataclasses
import logging
import math
import os
import sys
import urllib.request
import urllib.response
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, overload
from urllib.error import URLError

import torch
from torch import nn

from tabpfn.model.bar_distribution import FullSupportBarDistribution
from tabpfn.model.config import InferenceConfig
from tabpfn.model.encoders import (
    InputNormalizationEncoderStep,
    LinearInputEncoderStep,
    MulticlassClassificationTargetEncoder,
    NanHandlingEncoderStep,
    RemoveDuplicateFeaturesEncoderStep,
    RemoveEmptyFeaturesEncoderStep,
    SequentialEncoder,
    VariableNumFeaturesEncoderStep,
)
from tabpfn.model.transformer import PerFeatureTransformer

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"


class ModelVersion(str, Enum):
    V2 = "v2"


@dataclass
class ModelSource:
    repo_id: str
    default_filename: str
    filenames: list[str]

    @classmethod
    def get_classifier_v2(cls) -> ModelSource:
        filenames = [
            "tabpfn-v2-classifier.ckpt",
            "tabpfn-v2-classifier-gn2p4bpt.ckpt",
            "tabpfn-v2-classifier-llderlii.ckpt",
            "tabpfn-v2-classifier-od3j1g5m.ckpt",
            "tabpfn-v2-classifier-vutqq28w.ckpt",
            "tabpfn-v2-classifier-znskzxi4.ckpt",
        ]
        return cls(
            repo_id="Prior-Labs/TabPFN-v2-clf",
            default_filename="tabpfn-v2-classifier.ckpt",
            filenames=filenames,
        )

    @classmethod
    def get_regressor_v2(cls) -> ModelSource:
        filenames = [
            "tabpfn-v2-regressor.ckpt",
            "tabpfn-v2-regressor-09gpqh39.ckpt",
            "tabpfn-v2-regressor-2noar4o2.ckpt",
            "tabpfn-v2-regressor-5wof9ojf.ckpt",
            "tabpfn-v2-regressor-wyl4o83o.ckpt",
        ]
        return cls(
            repo_id="Prior-Labs/TabPFN-v2-reg",
            default_filename="tabpfn-v2-regressor.ckpt",
            filenames=filenames,
        )

    def get_fallback_urls(self) -> list[str]:
        return [
            f"https://huggingface.co/{self.repo_id}/resolve/main/{filename}?download=true"
            for filename in self.filenames
        ]


def _get_model_source(version: ModelVersion, model_type: ModelType) -> ModelSource:
    if version == ModelVersion.V2:
        if model_type == ModelType.CLASSIFIER:
            return ModelSource.get_classifier_v2()
        if model_type == ModelType.REGRESSOR:
            return ModelSource.get_regressor_v2()

    raise ValueError(
        f"Unsupported version/model combination: {version.value}/{model_type.value}",
    )


def _suppress_hf_token_warning():
    """Suppress warning about missing HuggingFace token."""
    import warnings

    # Filter warnings about HF_TOKEN
    warnings.filterwarnings(
        "ignore", message="The secret HF_TOKEN does not exist.*", category=UserWarning
    )


def _try_huggingface_downloads(
    base_path: Path,
    source: ModelSource,
    model_name: str | None = None,
    *,  # Force keyword-only arguments
    suppress_warnings: bool = True,
) -> None:
    """Try to download models using the HuggingFace Hub.

    Args:
        base_path: The path to save the downloaded model to.
        source: The source of the model.
        model_name: Optional specific model name to download.
        suppress_warnings: Whether to suppress HF token warnings.
    """
    if suppress_warnings:
        _suppress_hf_token_warning()
    """Try to download models and config using the HuggingFace Hub API."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "Please install huggingface_hub: pip install huggingface-hub",
        ) from e

    if model_name:
        if model_name not in source.filenames:
            raise ValueError(
                f"Model {model_name} not found in available models: {source.filenames}",
            )
        filename = model_name
    else:
        filename = source.default_filename
        if filename not in source.filenames:
            source.filenames.append(filename)

    logger.info(f"Attempting HuggingFace download: {filename}")

    # Create parent directory if it doesn't exist
    base_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Download model checkpoint
        local_path = hf_hub_download(
            repo_id=source.repo_id,
            filename=filename,
            local_dir=base_path.parent,
        )
        # Move model file to desired location
        Path(local_path).rename(base_path)

        # Download config.json
        try:
            config_path = base_path.parent / "config.json"
            config_local_path = hf_hub_download(
                repo_id=source.repo_id,
                filename="config.json",
                local_dir=base_path.parent,
            )
            if Path(config_local_path) != config_path:
                Path(config_local_path).rename(config_path)
            config_path.unlink()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to download config.json: {e!s}")
            # Continue even if config.json download fails

        logger.info(f"Successfully downloaded to {base_path}")
    except Exception as e:
        raise Exception("HuggingFace download failed!") from e


def _try_direct_downloads(
    base_path: Path,
    source: ModelSource,
    model_name: str | None = None,
) -> None:
    """Try to download models and config using direct URLs."""
    if model_name:
        if model_name not in source.filenames:
            raise ValueError(
                f"Model {model_name} not found in available models: {source.filenames}",
            )
        filename = model_name
    else:
        filename = source.default_filename
        if filename not in source.filenames:
            source.filenames.append(filename)

    model_url = (
        f"https://huggingface.co/{source.repo_id}/resolve/main/{filename}?download=true"
    )
    config_url = f"https://huggingface.co/{source.repo_id}/resolve/main/config.json?download=true"

    # Create parent directory if it doesn't exist
    base_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Attempting download from {model_url}")

    try:
        # Download model checkpoint
        with urllib.request.urlopen(model_url) as response:  # noqa: S310
            if response.status != 200:
                raise URLError(
                    f"HTTP {response.status} when downloading from {model_url}",
                )
            base_path.write_bytes(response.read())

        # Try to download config.json
        config_path = base_path.parent / "config.json"
        try:
            with urllib.request.urlopen(config_url) as response:  # noqa: S310
                if response.status == 200:
                    config_path.write_bytes(response.read())
        except Exception:  # noqa: BLE001
            logger.warning("Failed to download config.json!")
            # Continue even if config.json download fails

        logger.info(f"Successfully downloaded to {base_path}")
    except Exception as e:
        raise Exception("Direct download failed!") from e


def download_model(
    to: Path,
    *,
    version: Literal["v2"],
    which: Literal["classifier", "regressor"],
    model_name: str | None = None,
) -> Literal["ok"] | list[Exception]:
    """Download a TabPFN model, trying all available sources.

    Args:
        to: The directory to download the model to.
        version: The version of the model to download.
        which: The type of model to download.
        model_name: Optional specific model name to download.

    Returns:
        "ok" if the model was downloaded successfully, otherwise a list of
        exceptions that occurred that can be handled as desired.
    """
    errors: list[Exception] = []

    try:
        model_source = _get_model_source(ModelVersion(version), ModelType(which))
    except ValueError as e:
        return [e]

    try:
        _try_huggingface_downloads(to, model_source, model_name, suppress_warnings=True)
        return "ok"
    except Exception as e:  # noqa: BLE001
        logger.warning(f"HuggingFace downloads failed: {e!s}")
        errors.append(e)

    try:
        _try_direct_downloads(to, model_source, model_name)
        return "ok"
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Direct URL downloads failed: {e!s}")
        errors.append(e)

    return errors


def download_all_models(to: Path) -> None:
    """Download all v2 classifier and regressor models into a local directory."""
    to.mkdir(parents=True, exist_ok=True)
    for model_source, model_type in [
        (ModelSource.get_classifier_v2(), "classifier"),
        (ModelSource.get_regressor_v2(), "regressor"),
    ]:
        for ckpt_name in model_source.filenames:
            download_model(
                to=to / ckpt_name,
                version="v2",
                which=model_type,
                model_name=ckpt_name,
            )


def _user_cache_dir(platform: str, appname: str = "tabpfn") -> Path:
    use_instead_path = (Path.cwd() / ".tabpfn_models").resolve()

    # https://docs.python.org/3/library/sys.html#sys.platform
    if platform == "win32":
        # Honestly, I don't want to do what `platformdirs` does:
        # https://github.com/tox-dev/platformdirs/blob/b769439b2a3b70769a93905944a71b3e63ef4823/src/platformdirs/windows.py#L252-L265
        APPDATA_PATH = os.environ.get("APPDATA", "")
        if APPDATA_PATH.strip() != "":
            return Path(APPDATA_PATH) / appname

        warnings.warn(
            "Could not find APPDATA environment variable to get user cache dir,"
            " but detected platform 'win32'."
            f" Defaulting to a path '{use_instead_path}'."
            " If you would prefer, please specify a directory when creating"
            " the model.",
            UserWarning,
            stacklevel=2,
        )
        return use_instead_path

    if platform == "darwin":
        return Path.home() / "Library" / "Caches" / appname

    # TODO: Not entirely sure here, Python doesn't explicitly list
    # all of these and defaults to the underlying operating system
    # if not sure.
    linux_likes = ("freebsd", "linux", "netbsd", "openbsd")
    if any(platform.startswith(linux) for linux in linux_likes):
        # The reason to use "" as default is that the env var could exist but be empty.
        # We catch all this with the `.strip() != ""` below
        XDG_CACHE_HOME = os.environ.get("XDG_CACHE_HOME", "")
        if XDG_CACHE_HOME.strip() != "":
            return Path(XDG_CACHE_HOME) / appname
        return Path.home() / ".cache" / appname

    warnings.warn(
        f"Unknown platform '{platform}' to get user cache dir."
        f" Defaulting to a path at the execution site '{use_instead_path}'."
        " If you would prefer, please specify a directory when creating"
        " the model.",
        UserWarning,
        stacklevel=2,
    )
    return use_instead_path


@overload
def load_model_criterion_config(
    model_path: str | Path | None,
    *,
    check_bar_distribution_criterion: Literal[False],
    cache_trainset_representation: bool,
    version: Literal["v2"],
    which: Literal["classifier"],
    download: bool,
    model_seed: int,
) -> tuple[
    PerFeatureTransformer,
    nn.BCEWithLogitsLoss | nn.CrossEntropyLoss,
    InferenceConfig,
]: ...


@overload
def load_model_criterion_config(
    model_path: str | Path | None,
    *,
    check_bar_distribution_criterion: Literal[True],
    cache_trainset_representation: bool,
    version: Literal["v2"],
    which: Literal["regressor"],
    download: bool,
    model_seed: int,
) -> tuple[PerFeatureTransformer, FullSupportBarDistribution, InferenceConfig]: ...


def resolve_model_path(
    model_path: None | str | Path,
    which: Literal["regressor", "classifier"],
    version: Literal["v2"] = "v2",
) -> tuple[Path, Path, str, str]:
    if model_path is None:
        USER_TABPFN_CACHE_DIR_LOCATION = os.environ.get("TABPFN_MODEL_CACHE_DIR", "")
        if USER_TABPFN_CACHE_DIR_LOCATION.strip() != "":
            model_dir = Path(USER_TABPFN_CACHE_DIR_LOCATION)
        else:
            model_dir = _user_cache_dir(platform=sys.platform, appname="tabpfn")

        model_name = f"tabpfn-{version}-{which}.ckpt"
        model_path = model_dir / model_name
    else:
        if not isinstance(model_path, (str, Path)):
            raise ValueError(f"Invalid model_path: {model_path}")

        model_path = Path(model_path)
        model_dir = model_path.parent
        model_name = model_path.name

    return model_path, model_dir, model_name, which


def load_model_criterion_config(
    model_path: None | str | Path,
    *,
    check_bar_distribution_criterion: bool,
    cache_trainset_representation: bool,
    which: Literal["regressor", "classifier"],
    version: Literal["v2"] = "v2",
    download: bool,
    model_seed: int,
) -> tuple[
    PerFeatureTransformer,
    nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution,
    InferenceConfig,
]:
    """Load the model, criterion, and config from the given path.

    Args:
        model_path: The path to the model.
        check_bar_distribution_criterion:
            Whether to check if the criterion
            is a FullSupportBarDistribution, which is the expected criterion
            for models trained for regression.
        cache_trainset_representation:
            Whether the model should know to cache the trainset representation.
        which: Whether the model is a regressor or classifier.
        version: The version of the model.
        download: Whether to download the model if it doesn't exist.
        model_seed: The seed of the model.

    Returns:
        The model, criterion, and config.
    """
    (model_path, model_dir, model_name, which) = resolve_model_path(
        model_path, which, version
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        if not download:
            raise ValueError(
                f"Model path does not exist and downloading is disabled"
                f"\nmodel path: {model_path}",
            )

        # NOTE: We use warnings as:
        # * Logging is only visible if the user has logging enabled,
        #   which for the majority of people using Python, this is not
        #   the case.
        # * `print` has no way to easily be disabled from the outside.
        warnings.warn(
            f"Downloading model to {model_path}.",
            UserWarning,
            stacklevel=2,
        )
        res = download_model(
            model_path,
            version=version,
            which=which,
            model_name=model_name,
        )
        if res != "ok":
            repo_type = "clf" if which == "classifier" else "reg"
            raise RuntimeError(
                f"Failed to download model to {model_path}!\n\n"
                f"For offline usage, please download the model manually from:\n"
                f"https://huggingface.co/Prior-Labs/TabPFN-v2-{repo_type}/resolve/main/{model_name}\n\n"
                f"Then place it at: {model_path}",
            ) from res[0]

    loaded_model, criterion, config = load_model(path=model_path, model_seed=model_seed)
    loaded_model.cache_trainset_representation = cache_trainset_representation
    if check_bar_distribution_criterion and not isinstance(
        criterion,
        FullSupportBarDistribution,
    ):
        raise ValueError(
            f"The model loaded, '{model_path}', was expected to have a"
            " FullSupportBarDistribution criterion, but instead "
            f" had a {type(criterion).__name__} criterion.",
        )
    return loaded_model, criterion, config


def get_loss_criterion(
    config: InferenceConfig,
) -> nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution:
    # NOTE: We don't seem to have any of these
    if config.max_num_classes == 2:
        return nn.BCEWithLogitsLoss(reduction="none")

    if config.max_num_classes > 2:
        return nn.CrossEntropyLoss(reduction="none")

    assert config.max_num_classes == 0
    num_buckets = config.num_buckets

    # NOTE: This just seems to get overriddden in the module loading from `state_dict`
    # dummy values, extra bad s.t. one realizes if they are used for training
    borders = torch.arange(num_buckets + 1).float() * 10_000
    borders = borders * 3  # Used to be `config.get("bucket_scaling", 3)`

    return FullSupportBarDistribution(borders, ignore_nan_targets=True)


def _preprocess_config(config: dict) -> InferenceConfig:
    config["task_type"]
    batch_size = config["batch_size"]
    agg_k_grads = config.get("aggregate_k_gradients")

    if agg_k_grads is None:
        if not math.log(batch_size, 2).is_integer():
            raise ValueError(f"batch_size must be pow of 2, got {config['batch_size']}")

        second_dim_tokens = config.get("num_global_att_tokens ", config["seq_len"])
        memory_factor = (
            batch_size
            * config["nlayers"]
            * config["emsize"]
            * config["seq_len"]
            * second_dim_tokens
        )
        standard_memory_factor = 16 * 12 * 512 * 1200 * 1200
        agg_k_grads = math.ceil(memory_factor / (standard_memory_factor * 1.1))
        config["aggregate_k_gradients"] = agg_k_grads

        # Make sure that batch size is power of two
        config["batch_size"] = int(
            math.pow(2, math.floor(math.log(batch_size / agg_k_grads, 2))),
        )
        config["num_steps"] = math.ceil(config["num_steps"] * agg_k_grads)

        # Make sure that batch_size_per_gp_sample is power of two
        assert math.log(config["batch_size_per_gp_sample"], 2) % 1 == 0

    config.setdefault("recompute_attn", False)
    return InferenceConfig.from_dict(config)


def get_encoder(  # noqa: PLR0913
    *,
    num_features: int,
    embedding_size: int,
    remove_empty_features: bool,
    remove_duplicate_features: bool,
    nan_handling_enabled: bool,
    normalize_on_train_only: bool,
    normalize_to_ranking: bool,
    normalize_x: bool,
    remove_outliers: bool,
    normalize_by_used_features: bool,
    encoder_use_bias: bool,
) -> nn.Module:
    inputs_to_merge = {"main": {"dim": num_features}}

    encoder_steps = []
    if remove_empty_features:
        encoder_steps += [RemoveEmptyFeaturesEncoderStep()]

    if remove_duplicate_features:
        encoder_steps += [RemoveDuplicateFeaturesEncoderStep()]

    encoder_steps += [NanHandlingEncoderStep(keep_nans=nan_handling_enabled)]

    if nan_handling_enabled:
        inputs_to_merge["nan_indicators"] = {"dim": num_features}

        encoder_steps += [
            VariableNumFeaturesEncoderStep(
                num_features=num_features,
                normalize_by_used_features=False,
                in_keys=["nan_indicators"],
                out_keys=["nan_indicators"],
            ),
        ]

    encoder_steps += [
        InputNormalizationEncoderStep(
            normalize_on_train_only=normalize_on_train_only,
            normalize_to_ranking=normalize_to_ranking,
            normalize_x=normalize_x,
            remove_outliers=remove_outliers,
        ),
    ]

    encoder_steps += [
        VariableNumFeaturesEncoderStep(
            num_features=num_features,
            normalize_by_used_features=normalize_by_used_features,
        ),
    ]

    encoder_steps += [
        LinearInputEncoderStep(
            num_features=sum([i["dim"] for i in inputs_to_merge.values()]),
            emsize=embedding_size,
            bias=encoder_use_bias,
            in_keys=tuple(inputs_to_merge),
            out_keys=("output",),
        ),
    ]

    return SequentialEncoder(*encoder_steps, output_key="output")


def get_y_encoder(
    *,
    num_inputs: int,
    embedding_size: int,
    nan_handling_y_encoder: bool,
    max_num_classes: int,
) -> nn.Module:
    steps = []
    inputs_to_merge = [{"name": "main", "dim": num_inputs}]
    if nan_handling_y_encoder:
        steps += [NanHandlingEncoderStep()]
        inputs_to_merge += [{"name": "nan_indicators", "dim": num_inputs}]

    if max_num_classes >= 2:
        steps += [MulticlassClassificationTargetEncoder()]

    steps += [
        LinearInputEncoderStep(
            num_features=sum([i["dim"] for i in inputs_to_merge]),  # type: ignore
            emsize=embedding_size,
            in_keys=tuple(i["name"] for i in inputs_to_merge),  # type: ignore
            out_keys=("output",),
        ),
    ]
    return SequentialEncoder(*steps, output_key="output")


def load_model(
    *,
    path: Path,
    model_seed: int,
) -> tuple[
    PerFeatureTransformer,
    nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution,
    InferenceConfig,
]:
    """Loads a model from a given path.

    Args:
        path: Path to the checkpoint
        model_seed: The seed to use for the model
    """
    # Catch the `FutureWarning` that torch raises. This should be dealt with!
    # The warning is raised due to `torch.load`, which advises against ckpt
    # files that contain non-tensor data.
    # This `weightes_only=None` is the default value. In the future this will default to
    # `True`, dissallowing loading of arbitrary objects.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        checkpoint = torch.load(path, map_location="cpu", weights_only=None)

    assert "state_dict" in checkpoint
    assert "config" in checkpoint

    state_dict = checkpoint["state_dict"]
    config = _preprocess_config(checkpoint["config"])

    criterion_state_keys = [k for k in state_dict if "criterion." in k]
    loss_criterion = get_loss_criterion(config)
    if isinstance(loss_criterion, FullSupportBarDistribution):
        # Remove from state dict
        criterion_state = {
            k.replace("criterion.", ""): state_dict.pop(k) for k in criterion_state_keys
        }
        loss_criterion.load_state_dict(criterion_state)
    else:
        assert len(criterion_state_keys) == 0, criterion_state_keys

    # Old decision tree for n_out, made equivalent:
    # > if config.max_num_classes == 2:
    # >   loss = Losses.bce           (n_out -> 1)
    # > elif config.max_num_classes > 2:
    # >   loss = Losses.ce            (n_out -> config.max_num_classes)
    # > else:
    # >   create_bar_distribution ... (n_out -> loss.num_bars)
    #
    # > if loss is bardist:
    # >   n_out = loss.num_bars
    # > elif loss is CrossEntropyLoss:
    # >   n_out = config.max_num_classes
    # > else:
    # >  n_out = 1
    n_out: int
    if config.max_num_classes == 2:
        n_out = 1
    elif config.max_num_classes > 2:
        n_out = config.max_num_classes
    else:
        assert config.max_num_classes == 0
        assert isinstance(loss_criterion, FullSupportBarDistribution)
        n_out = loss_criterion.num_bars

    model = PerFeatureTransformer(
        seed=model_seed,
        # Things that were explicitly passed inside `build_model()`
        encoder=get_encoder(
            num_features=config.features_per_group,
            embedding_size=config.emsize,
            remove_empty_features=config.remove_empty_features,
            remove_duplicate_features=config.remove_duplicate_features,
            nan_handling_enabled=config.nan_handling_enabled,
            normalize_on_train_only=config.normalize_on_train_only,
            normalize_to_ranking=config.normalize_to_ranking,
            normalize_x=config.normalize_x,
            remove_outliers=config.remove_outliers,
            normalize_by_used_features=config.normalize_by_used_features,
            encoder_use_bias=config.encoder_use_bias,
        ),
        y_encoder=get_y_encoder(
            num_inputs=1,
            embedding_size=config.emsize,
            nan_handling_y_encoder=config.nan_handling_y_encoder,
            max_num_classes=config.max_num_classes,
        ),
        nhead=config.nhead,
        ninp=config.emsize,
        nhid=config.emsize * config.nhid_factor,
        nlayers=config.nlayers,
        features_per_group=config.features_per_group,
        cache_trainset_representation=True,
        #
        # Based on not being present in config or otherwise, these were default values
        init_method=None,
        decoder_dict={"standard": (None, n_out)},
        use_encoder_compression_layer=False,
        #
        # These were extra things passed in through `**model_extra_args`
        # or `**extra_model_kwargs` and were present in the config
        recompute_attn=config.recompute_attn,
        recompute_layer=config.recompute_layer,
        feature_positional_embedding=config.feature_positional_embedding,
        use_separate_decoder=config.use_separate_decoder,
        #
        # These are things that had default values from config.get() but were not
        # present in any config.
        layer_norm_with_elementwise_affine=False,
        nlayers_decoder=None,
        pre_norm=False,
        #
        # These seem to map to `**layer_config` in the init of `PerFeatureTransformer`
        # Which got passed to the `PerFeatureEncoderLayer(**layer_config)`
        multiquery_item_attention=config.multiquery_item_attention,  # False
        multiquery_item_attention_for_test_set=config.multiquery_item_attention_for_test_set,  # True  # noqa: E501
        # Is either 1.0 or None in the configs, which lead to the default of 1.0 anywho
        attention_init_gain=(
            config.attention_init_gain
            if config.attention_init_gain is not None
            else 1.0
        ),
        # Is True, False in the config or not present,
        # with the default of the `PerFeatureEncoderLayer` being False,
        # which is what the value would have mapped to if the config had not present
        two_sets_of_queries=(
            config.two_sets_of_queries
            if config.two_sets_of_queries is not None
            else False
        ),
    )

    model.load_state_dict(state_dict)
    model.eval()
    return model, loss_criterion, config


# NOTE: This function doesn't seem to be used anywhere.
def save_tabpfn_model(model: nn.Module, save_path: Path | str) -> None:
    # Get model state dict
    model_state = model.model_.state_dict()

    # Get bardist state dict and prefix with 'criterion.'
    if hasattr(model, "bardist_") and model.bardist_ is not None:
        bardist_state = {
            f"criterion.{k}": v for k, v in model.bardist_.state_dict().items()
        }
        # Combine model and bardist states
        state_dict = {**model_state, **bardist_state}
    else:
        state_dict = model_state

    # Convert Config object to dictionary and add necessary fields
    config_dict = dataclasses.asdict(model.config_)

    # Create checkpoint with correct structure
    checkpoint = {"state_dict": state_dict, "config": config_dict}

    # Save the checkpoint
    torch.save(checkpoint, save_path)
