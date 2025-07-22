#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import sys
import tempfile
import urllib.request
import urllib.response
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, overload
from urllib.error import URLError

import joblib
import torch
from torch import nn

from tabpfn.inference import InferenceEngine
from tabpfn.model.bar_distribution import BarDistribution, FullSupportBarDistribution
from tabpfn.model.config import ModelConfig
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

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

    from tabpfn import TabPFNClassifier, TabPFNRegressor

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
            "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
        ]
        return cls(
            repo_id="Prior-Labs/TabPFN-v2-clf",
            default_filename="tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
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

    warning_context = (
        warnings.catch_warnings() if suppress_warnings else contextlib.nullcontext()
    )

    with warning_context:
        if suppress_warnings:
            warnings.filterwarnings("ignore")

        try:
            # Download model checkpoint
            local_path = hf_hub_download(
                repo_id=source.repo_id,
                filename=filename,
                local_dir=base_path.parent,
            )
            # Move model file to desired location
            Path(local_path).rename(base_path)

            # Download config.json only to increment the download counter. We do not
            # actually use this file so it is removed immediately after download.
            try:
                config_local_path = hf_hub_download(
                    repo_id=source.repo_id,
                    filename="config.json",
                    local_dir=base_path.parent,
                )
                Path(config_local_path).unlink(missing_ok=True)
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
                which=cast("Literal['classifier', 'regressor']", model_type),
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
) -> tuple[
    PerFeatureTransformer,
    nn.BCEWithLogitsLoss | nn.CrossEntropyLoss,
    ModelConfig,
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
) -> tuple[PerFeatureTransformer, FullSupportBarDistribution, ModelConfig]: ...


def load_model_criterion_config(
    model_path: None | str | Path,
    *,
    check_bar_distribution_criterion: bool,
    cache_trainset_representation: bool,
    which: Literal["regressor", "classifier"],
    version: Literal["v2"] = "v2",
    download: bool,
) -> tuple[
    PerFeatureTransformer,
    nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution,
    ModelConfig,
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

        logger.info(f"Downloading model to {model_path}.")
        res = download_model(
            model_path,
            version=version,
            which=cast("Literal['classifier', 'regressor']", which),
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

    loaded_model, criterion, config = load_model(path=model_path)
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


def resolve_model_path(
    model_path: None | str | Path,
    which: Literal["regressor", "classifier"],
    version: Literal["v2"] = "v2",
) -> tuple[Path, Path, str, str]:
    """Resolves the model path, using the official default model if no path is provided.

    Args:
        model_path: An optional path to a model file. If None, the default
            model for the given `which` and `version` will be used, resolving
            to the local cache directory.
        which: The type of model ('regressor' or 'classifier').
        version: The model version (currently only 'v2').

    Returns:
        A tuple containing the resolved model Path, the parent directory Path,
        the model's filename, and the model type.
    """
    if model_path is None:
        # Get the source information to find the official default model filename.
        model_source = _get_model_source(ModelVersion(version), ModelType(which))
        model_name = model_source.default_filename

        # Determine the cache directory for storing models.
        USER_TABPFN_CACHE_DIR_LOCATION = os.environ.get("TABPFN_MODEL_CACHE_DIR", "")
        if USER_TABPFN_CACHE_DIR_LOCATION.strip() != "":
            model_dir = Path(USER_TABPFN_CACHE_DIR_LOCATION)
        else:
            model_dir = _user_cache_dir(platform=sys.platform, appname="tabpfn")

        # Construct the full path to the default model.
        model_path = model_dir / model_name
    else:
        # If a path is provided, simply parse it.
        if not isinstance(model_path, (str, Path)):
            raise ValueError(f"Invalid model_path: {model_path}")

        model_path = Path(model_path)
        model_dir = model_path.parent
        model_name = model_path.name

    return model_path, model_dir, model_name, which


def get_loss_criterion(
    config: ModelConfig,
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


def load_model_from_config(
    *,
    config: ModelConfig,
    loss_criterion: nn.BCEWithLogitsLoss
    | nn.CrossEntropyLoss
    | FullSupportBarDistribution,
    load_for_inference: bool = True,
) -> PerFeatureTransformer:
    """Loads a model from a given config.

    Args:
        config: The config to load the model from.
        loss_criterion: The loss function object created from the given config.
        load_for_inference: Whether to load the model for inference. Controls whether
            the model is set to evaluation mode and whether the trainset representation
            is cached.
    """
    model = PerFeatureTransformer(
        config=config,
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
        cache_trainset_representation=load_for_inference,
        use_encoder_compression_layer=False,
        n_out=get_n_out(config, loss_criterion),
        #
        # These were extra things passed in through `**model_extra_args`
        # or `**extra_model_kwargs` and were present in the config
        feature_positional_embedding=config.feature_positional_embedding,
        #
        # These are things that had default values from config.get() but were not
        # present in any config.
        layer_norm_with_elementwise_affine=False,
    )
    if load_for_inference:
        model.eval()
    return model


def load_model(
    *,
    path: Path,
) -> tuple[
    PerFeatureTransformer,
    nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution,
    ModelConfig,
]:
    """Loads a model from a given path. Only for inference.

    Args:
        path: Path to the checkpoint
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
    config = ModelConfig.from_dict(ModelConfig.upgrade_config(checkpoint["config"]))

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

    model = load_model_from_config(config=config, loss_criterion=loss_criterion)
    model.load_state_dict(state_dict)
    model.eval()

    return model, loss_criterion, config


def get_n_out(
    config: ModelConfig,
    loss: nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution,
) -> int:
    """Works out the number of outputs of the model."""
    if config.max_num_classes == 2:
        return 1
    if config.max_num_classes > 2 and isinstance(loss, nn.CrossEntropyLoss):
        return config.max_num_classes
    if config.max_num_classes == 0 and isinstance(loss, BarDistribution):
        return loss.num_bars
    raise ValueError(
        "Unknown configuration: "
        f"max_num_classes={config.max_num_classes} and loss={type(loss)}"
    )


def save_tabpfn_model(
    model: TabPFNRegressor | TabPFNClassifier, save_path: Path | str
) -> None:
    """Save the underlying TabPFN foundation model to ``save_path``.

    This writes only the base pre-trained weights and configuration. It does
    **not** store a fitted :class:`TabPFNRegressor`/``Classifier`` instance.
    The resulting file is merely a checkpoint consumed by
    :func:`load_model_criterion_config` to build a new estimator.

    Args:
        model:
            The internal model object of a ``TabPFN`` estimator.
        save_path:
            Path to save the checkpoint to.
    """
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

    # Create checkpoint with correct structure
    checkpoint = {"state_dict": state_dict, "config": model.config_}

    # Save the checkpoint
    torch.save(checkpoint, save_path)


def save_fitted_tabpfn_model(estimator: BaseEstimator, path: Path | str) -> None:
    """Persist a fitted TabPFN estimator to ``path``.

    This stores the initialization parameters and the fitted state, but crucially
    omits the large foundation model weights for efficiency.
    """
    if not hasattr(estimator, "executor_"):
        raise RuntimeError("Estimator must be fitted before saving.")

    path = Path(path)
    if path.suffix != ".tabpfn_fit":
        raise ValueError("Path must end with .tabpfn_fit")

    # Attributes that are handled separately or should not be saved.
    blacklist = {"model_", "executor_", "config_", "device_"}

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. Save init parameters to JSON
        params = estimator.get_params(deep=False)
        params = {
            k: (str(v) if isinstance(v, torch.dtype) else v) for k, v in params.items()
        }
        params["__class_name__"] = estimator.__class__.__name__
        with (tmp / "init_params.json").open("w") as f:
            json.dump(params, f)

        # 2. Automatically save all scikit-learn fitted attributes
        fitted_attrs = {
            key: value
            for key, value in vars(estimator).items()
            if key.endswith("_") and key not in blacklist
        }
        joblib.dump(fitted_attrs, tmp / "fitted_attrs.joblib")

        # 3. Save the InferenceEngine state without the model weights
        estimator.executor_.save_state_expect_model_weights(
            tmp / "executor_state.joblib"
        )

        # 4. Create the final zip archive
        shutil.make_archive(str(path).replace(".tabpfn_fit", ""), "zip", tmp)
        shutil.move(str(path).replace(".tabpfn_fit", "") + ".zip", path)


def _extract_archive(path: Path, tmp: Path) -> None:
    import zipfile

    with zipfile.ZipFile(path, "r") as archive:
        for member in archive.namelist():
            member_path = (tmp / member).resolve()
            if not str(member_path).startswith(str(tmp.resolve())):
                raise ValueError(f"Unsafe file path detected: {member}")
            archive.extract(member, tmp)


def load_fitted_tabpfn_model(
    path: Path | str, *, device: str | torch.device = "cpu"
) -> BaseEstimator:
    """Load a fitted TabPFN estimator saved with ``save_fitted_tabpfn_model``."""
    from copy import deepcopy
    from importlib import import_module

    path = Path(path)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Extract the archive to a temporary directory
        _extract_archive(path, tmp)

        # 1. Load init params and create a fresh estimator instance
        with (tmp / "init_params.json").open() as f:
            params = json.load(f)

        saved_cls_name = params.pop("__class_name__")
        if isinstance(params.get("inference_precision"), str) and params[
            "inference_precision"
        ].startswith("torch."):
            dtype_name = params["inference_precision"].split(".")[1]
            params["inference_precision"] = getattr(torch, dtype_name)
        params["device"] = device

        if saved_cls_name == "TabPFNClassifier":
            cls = import_module("tabpfn.classifier").TabPFNClassifier
        elif saved_cls_name == "TabPFNRegressor":
            cls = import_module("tabpfn.regressor").TabPFNRegressor
        else:
            raise TypeError(f"Unknown estimator class '{saved_cls_name}'")

        est = cls(**params)
        # This is critical: it loads the base model weights into `est.model_`
        est._initialize_model_variables()

        # 2. Restore all other fitted attributes
        fitted_attrs = joblib.load(tmp / "fitted_attrs.joblib")
        for key, value in fitted_attrs.items():
            setattr(est, key, value)

        # 3. Load the InferenceEngine state
        est.executor_ = InferenceEngine.load_state(tmp / "executor_state.joblib")

        # 4. Re-link the foundation model with the loaded engine
        if hasattr(est.executor_, "model") and est.executor_.model is None:
            est.executor_.model = est.model_

        if hasattr(est.executor_, "models") and est.executor_.models is None:
            est.executor_.models = [
                deepcopy(est.model_) for _ in range(len(est.executor_.ensemble_configs))
            ]

        # 5. Move all torch components to the target device
        est.device_ = torch.device(device)
        if hasattr(est.executor_, "model") and est.executor_.model is not None:
            est.executor_.model.to(est.device_)
        if hasattr(est.executor_, "models"):
            est.executor_.models = [m.to(est.device_) for m in est.executor_.models]

        # Restore other potential torch objects from fitted_attrs
        for key, value in vars(est).items():
            if key.endswith("_") and hasattr(value, "to"):
                setattr(est, key, value.to(est.device_))

        return est
