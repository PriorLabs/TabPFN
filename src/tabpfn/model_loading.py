"""Functions for downloading and loading model checkpoints."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import os
import shutil
import sys
import tempfile
import urllib.request
import warnings
import zipfile
from dataclasses import asdict, dataclass
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union, cast, overload
from urllib.error import URLError

import joblib
import torch
from tabpfn_common_utils.telemetry import set_model_config
from torch import nn

from tabpfn.architectures import ARCHITECTURES
from tabpfn.architectures.base.bar_distribution import (
    BarDistribution,
    FullSupportBarDistribution,
)
from tabpfn.constants import ModelVersion
from tabpfn.inference import InferenceEngine
from tabpfn.inference_config import InferenceConfig
from tabpfn.settings import settings

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

    from tabpfn import TabPFNClassifier, TabPFNRegressor

if TYPE_CHECKING:
    from tabpfn.architectures.interface import Architecture, ArchitectureConfig
    from tabpfn.constants import ModelPath

logger = logging.getLogger(__name__)

# Public fallback base URL for model downloads
FALLBACK_S3_BASE_URL = "https://storage.googleapis.com/tabpfn-v2-model-files/05152025"

# Special string used to identify v2.5 models in model paths.
V_2_5_IDENTIFIER = "v2.5"


class ModelType(str, Enum):  # noqa: D101
    # TODO: Merge with TaskType in tabpfn.constants.
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"


@dataclass
class ModelSource:  # noqa: D101
    repo_id: str
    default_filename: str
    filenames: list[str]

    @classmethod
    def get_classifier_v2(cls) -> ModelSource:  # noqa: D102
        filenames = [
            "tabpfn-v2-classifier.ckpt",
            "tabpfn-v2-classifier-gn2p4bpt.ckpt",
            "tabpfn-v2-classifier-llderlii.ckpt",
            "tabpfn-v2-classifier-od3j1g5m.ckpt",
            "tabpfn-v2-classifier-vutqq28w.ckpt",
            "tabpfn-v2-classifier-znskzxi4.ckpt",
            "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
            "tabpfn-v2-classifier-finetuned-znskzxi4-tvvss6bp.ckpt",
            "tabpfn-v2-classifier-finetuned-vutqq28w-boexhu6h.ckpt",
            "tabpfn-v2-classifier-finetuned-od3j1g5m-4svepuy5.ckpt",
            "tabpfn-v2-classifier-finetuned-llderlii-oyd7ul21.ckpt",
            "tabpfn-v2-classifier-finetuned-gn2p4bpt-xp6f0iqb.ckpt",
            "tabpfn-v2-classifier-v2_default.ckpt",
        ]
        return cls(
            repo_id="Prior-Labs/TabPFN-v2-clf",
            default_filename="tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
            filenames=filenames,
        )

    @classmethod
    def get_regressor_v2(cls) -> ModelSource:  # noqa: D102
        filenames = [
            "tabpfn-v2-regressor.ckpt",
            "tabpfn-v2-regressor-09gpqh39.ckpt",
            "tabpfn-v2-regressor-2noar4o2.ckpt",
            "tabpfn-v2-regressor-wyl4o83o.ckpt",
            "tabpfn-v2-regressor-v2_default.ckpt",
        ]
        return cls(
            repo_id="Prior-Labs/TabPFN-v2-reg",
            default_filename="tabpfn-v2-regressor.ckpt",
            filenames=filenames,
        )

    @classmethod
    def get_classifier_v2_5(cls) -> ModelSource:  # noqa: D102
        filenames = [
            "tabpfn-v2.5-classifier-v2.5_default.ckpt",
            "tabpfn-v2.5-classifier-v2.5_default-2.ckpt",
            "tabpfn-v2.5-classifier-v2.5_large-features-L.ckpt",
            "tabpfn-v2.5-classifier-v2.5_large-features-XL.ckpt",
            "tabpfn-v2.5-classifier-v2.5_large-samples.ckpt",
            "tabpfn-v2.5-classifier-v2.5_real-large-features.ckpt",
            "tabpfn-v2.5-classifier-v2.5_real-large-samples-and-features.ckpt",
            "tabpfn-v2.5-classifier-v2.5_real.ckpt",
            "tabpfn-v2.5-classifier-v2.5_variant.ckpt",
        ]
        return cls(
            repo_id="Prior-Labs/tabpfn_2_5",
            default_filename="tabpfn-v2.5-classifier-v2.5_default.ckpt",
            filenames=filenames,
        )

    @classmethod
    def get_regressor_v2_5(cls) -> ModelSource:  # noqa: D102
        filenames = [
            "tabpfn-v2.5-regressor-v2.5_default.ckpt",
            "tabpfn-v2.5-regressor-v2.5_low-skew.ckpt",
            "tabpfn-v2.5-regressor-v2.5_quantiles.ckpt",
            "tabpfn-v2.5-regressor-v2.5_real-variant.ckpt",
            "tabpfn-v2.5-regressor-v2.5_real.ckpt",
            "tabpfn-v2.5-regressor-v2.5_small-samples.ckpt",
            "tabpfn-v2.5-regressor-v2.5_variant.ckpt",
        ]
        return cls(
            repo_id="Prior-Labs/tabpfn_2_5",
            default_filename="tabpfn-v2.5-regressor-v2.5_default.ckpt",
            filenames=filenames,
        )


def _get_model_source(version: ModelVersion, model_type: ModelType) -> ModelSource:
    if version == ModelVersion.V2:
        if model_type == ModelType.CLASSIFIER:
            return ModelSource.get_classifier_v2()
        if model_type == ModelType.REGRESSOR:
            return ModelSource.get_regressor_v2()
    elif version == ModelVersion.V2_5:
        if model_type == ModelType.CLASSIFIER:
            return ModelSource.get_classifier_v2_5()
        if model_type == ModelType.REGRESSOR:
            return ModelSource.get_regressor_v2_5()

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
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        # Import specific exceptions for better error handling
        from huggingface_hub.utils import (  # noqa: PLC0415
            GatedRepoError,
            HfHubHTTPError,
        )
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
            should_download_config = not base_path.exists()

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
            # Note that we also handle model caching ourselves, so we don't double
            # count, even with removing the config.json afterwards.
            try:
                if should_download_config:
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

        except (GatedRepoError, HfHubHTTPError) as e:
            # Check if this is an authentication/gating error
            is_auth_error = False
            if isinstance(e, GatedRepoError) or (
                isinstance(e, HfHubHTTPError) and e.response.status_code in (401, 403)
            ):
                is_auth_error = True

            if is_auth_error:
                auth_message = (
                    f"Authentication error downloading from '{source.repo_id}'.\n"
                    "This model is gated and requires you to accept its terms.\n\n"
                    "Please follow these steps:\n"
                    f"1. Visit https://huggingface.co/{source.repo_id} in your "
                    f"browser and"
                    f" accept the terms of use.\n"
                    "2. Log in to your Hugging Face account via"
                    " the command line by running:\n"
                    "   hf auth login\n"
                    "(Alternatively, you can set the HF_TOKEN environment variable"
                    " with a read token).\n\n"
                    "For detailed instructions, see "
                    "https://docs.priorlabs.ai/how-to-access-gated-models"
                )
                raise RuntimeError(auth_message)  # noqa: B904
            raise e


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

    url_pairs = [
        (
            f"https://huggingface.co/{source.repo_id}/resolve/main/{filename}?download=true",
            f"https://huggingface.co/{source.repo_id}/resolve/main/config.json?download=true",
        ),
        (f"{FALLBACK_S3_BASE_URL}/{filename}", f"{FALLBACK_S3_BASE_URL}/config.json"),
    ]

    last_error: Exception | None = None
    base_path.parent.mkdir(parents=True, exist_ok=True)

    for model_url, config_url in url_pairs:
        logger.info(f"Attempting download from {model_url}")
        try:
            with urllib.request.urlopen(model_url, timeout=180) as response:  # noqa: S310
                if response.status != 200:
                    raise URLError(
                        f"HTTP {response.status} when downloading from {model_url}",
                    )
                base_path.write_bytes(response.read())

            config_path = base_path.parent / "config.json"
            try:
                with urllib.request.urlopen(config_url, timeout=180) as response:  # noqa: S310
                    if response.status == 200:
                        config_path.write_bytes(response.read())
            except Exception:  # noqa: BLE001
                logger.warning("Failed to download config.json!")

            logger.info(f"Successfully downloaded to {base_path}")
            return
        except Exception as e:  # noqa: BLE001
            last_error = e
            logger.warning(f"Failed download from {model_url}: {e!s}")

    raise Exception("Direct download failed!") from last_error


def download_model(
    to: Path,
    *,
    version: ModelVersion,
    which: Literal["classifier", "regressor"],
    model_name: str | None = None,
) -> Literal["ok"] | list[Exception]:
    """Download a TabPFN model, trying all available sources.

    Args:
        to: The file path to download the model to.
        version: The version of the model to download.
        which: The type of model to download.
        model_name: Optional specific model name to download.

    Returns:
        "ok" if the model was downloaded successfully, otherwise a list of
        exceptions that occurred that can be handled as desired.
    """
    errors: list[Exception] = []

    try:
        model_source = _get_model_source(version, ModelType(which))
    except ValueError as e:
        return [e]

    try:
        _try_huggingface_downloads(to, model_source, model_name, suppress_warnings=True)
        return "ok"
    except Exception as e:  # noqa: BLE001
        logger.warning("HuggingFace download failed.")
        errors.append(e)

    # For Version 2.5 we require gating, which we don't have in place for direct
    # downloads.
    if version == ModelVersion.V2:
        try:
            _try_direct_downloads(to, model_source, model_name)
            return "ok"
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Direct URL downloads failed: {e!s}")
            errors.append(e)
    else:
        logger.warning(
            "For commercial usage, we provide alternative download options for v2.5, "
            "please reach out to us at sales@priorlabs.ai."
        )

    return errors


def download_all_models(to: Path) -> None:
    """Download all available classifier and regressor models into a local directory."""
    to.mkdir(parents=True, exist_ok=True)
    for model_version, model_source, model_type in [
        (ModelVersion.V2, ModelSource.get_classifier_v2(), "classifier"),
        (ModelVersion.V2, ModelSource.get_regressor_v2(), "regressor"),
        (ModelVersion.V2_5, ModelSource.get_classifier_v2_5(), "classifier"),
        (ModelVersion.V2_5, ModelSource.get_regressor_v2_5(), "regressor"),
    ]:
        for ckpt_name in model_source.filenames:
            path = to / ckpt_name
            if path.exists():
                logger.info(
                    f"Skipping download of checkpoint that already exists: {path}"
                )
                continue
            download_model(
                to=path,
                version=model_version,
                which=cast("Literal['classifier', 'regressor']", model_type),
                model_name=ckpt_name,
            )


def get_cache_dir() -> Path:  # noqa: PLR0911
    """Get the cache directory for TabPFN models, as appropriate for the platform."""
    if settings.tabpfn.model_cache_dir is not None:
        return settings.tabpfn.model_cache_dir

    platform = sys.platform
    appname = "tabpfn"
    use_instead_path = (Path.cwd() / ".tabpfn_models").resolve()

    if platform == "win32":
        # Do something similar to platformdirs, but very simplified:
        # https://github.com/tox-dev/platformdirs/blob/b769439b2a3b70769a93905944a71b3e63ef4823/src/platformdirs/windows.py#L252-L265
        # Unclear how well this works.
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


P = TypeVar("P", bound=Union[str, list[str]])


def prepend_cache_path(model_path: P) -> P:
    """Prepends the TabPFN cache directory to the given path or paths.

    The cache directory is selected appropriately for the platform.
    """
    cache_dir = get_cache_dir()
    if isinstance(model_path, list):
        return [str(cache_dir / path) for path in model_path]
    return str(cache_dir / model_path)


@overload
def load_model_criterion_config(
    model_path: ModelPath | list[ModelPath] | None,
    *,
    check_bar_distribution_criterion: Literal[False],
    cache_trainset_representation: bool,
    version: Literal["v2", "v2.5"],
    which: Literal["classifier"],
    download_if_not_exists: bool,
) -> tuple[
    list[Architecture],
    nn.BCEWithLogitsLoss | nn.CrossEntropyLoss,
    list[ArchitectureConfig],
    InferenceConfig,
]: ...


@overload
def load_model_criterion_config(
    model_path: ModelPath | list[ModelPath] | None,
    *,
    check_bar_distribution_criterion: Literal[True],
    cache_trainset_representation: bool,
    version: Literal["v2", "v2.5"],
    which: Literal["regressor"],
    download_if_not_exists: bool,
) -> tuple[
    list[Architecture],
    FullSupportBarDistribution,
    list[ArchitectureConfig],
    InferenceConfig,
]: ...


def load_model_criterion_config(
    model_path: ModelPath | list[ModelPath] | None,
    *,
    check_bar_distribution_criterion: bool,
    cache_trainset_representation: bool,
    which: Literal["regressor", "classifier"],
    version: Literal["v2", "v2.5"] = "v2",
    download_if_not_exists: bool,
) -> tuple[
    list[Architecture],
    nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution,
    list[ArchitectureConfig],
    InferenceConfig,
]:
    """Load the model(s), criterion(s), and config(s) from the given path.

    If multiple model paths are provided, then all models must use the same criterion
    and inference config.

    Args:
        model_path: The path to the model, or list of paths if multiple models should be
            loaded.
        check_bar_distribution_criterion:
            Whether to check if the criterion
            is a FullSupportBarDistribution, which is the expected criterion
            for models trained for regression.
        cache_trainset_representation:
            Whether the model should know to cache the trainset representation.
        which: Whether the model is a regressor or classifier.
        version: The version of the model.
        download_if_not_exists: Whether to download the model if it doesn't exist.

    Returns:
        list of models, the criterion, list of architecture configs, the inference
        config
    """
    model_version = ModelVersion(version)
    (resolved_model_paths, resolved_model_dirs, resolved_model_names, which) = (
        resolve_model_path(
            model_path=model_path,
            which=which,
            version=model_version.value,
        )
    )

    # Anonymously track the model config for usage telemetry
    _log_model_config(resolved_model_paths, which, model_version)

    for folder in resolved_model_dirs:
        folder.mkdir(parents=True, exist_ok=True)

    loaded_models = []
    criterions = []
    architecture_configs = []
    inference_configs = []

    for i, path in enumerate(resolved_model_paths):
        if not path.exists():
            if not download_if_not_exists:
                raise ValueError(
                    f"Model path does not exist and downloading is disabled"
                    f"\nmodel path: {path}",
                )

            logger.info(f"Downloading model to {path}.")
            res = download_model(
                path,
                version=model_version,
                which=cast("Literal['classifier', 'regressor']", which),
                model_name=resolved_model_names[i],
            )
            if res != "ok":
                # Later: Add improved error handling here, reenabling
                #  the old offline download (only raise when Gating)
                raise res[0]

        loaded_model, criterion, architecture_config, inference_config = load_model(
            path=path,
            cache_trainset_representation=cache_trainset_representation,
        )
        if check_bar_distribution_criterion and not isinstance(
            criterion,
            FullSupportBarDistribution,
        ):
            raise ValueError(
                f"The model loaded, '{path}', was expected to have a"
                " FullSupportBarDistribution criterion, but instead "
                f" had a {type(criterion).__name__} criterion.",
            )
        loaded_models.append(loaded_model)
        criterions.append(criterion)
        architecture_configs.append(architecture_config)
        inference_configs.append(inference_config)

    first_criterion = criterions[0]
    if isinstance(first_criterion, FullSupportBarDistribution):
        for criterion in criterions[1:]:
            if not first_criterion.has_equal_borders(criterion):
                raise ValueError(
                    f"The criterions {first_criterion} and {criterion} are different. "
                    "This is not supported in the current implementation"
                )

    first_inference_config = inference_configs[0]
    for inference_config in inference_configs[1:]:
        if inference_config != first_inference_config:
            raise ValueError(
                f"Config 1: {first_inference_config}\n"
                f"Config 2: {inference_config}\n"
                "Inference configs for different models are different, which is not "
                "supported. See above."
            )

    return loaded_models, first_criterion, architecture_configs, first_inference_config


def _resolve_model_version(model_path: ModelPath | None) -> ModelVersion:
    if model_path is None:
        return settings.tabpfn.model_version
    if V_2_5_IDENTIFIER in Path(model_path).name:
        return ModelVersion.V2_5
    return ModelVersion.V2


def _log_model_config(
    model_paths: list[Path],
    which: Literal["classifier", "regressor"],
    version: ModelVersion,
) -> None:
    """Set the model config (model_path and model_version) for anonymous
    usage telemetry.

    Args:
        model_paths: The path(s) to the model.
        which: The type of model ('classifier' or 'regressor').
        version: The model version (currently only 'v2' or 'v2.5').
    """
    if len(model_paths) != 1:
        return

    model_type = ModelType(which)
    model_source = _get_model_source(version, model_type)

    path: Path = model_paths[0]
    # Check to avoid that we pass in arbitrary paths containing e.g. PII
    # Ensure we whitelist model names so that no PII can be released.
    if path.name in model_source.filenames:
        set_model_config(path.name, version.value)
    else:
        set_model_config("OTHER", version.value)


def log_model_init_params(
    estimator: TabPFNClassifier | TabPFNRegressor, params: dict[str, Any]
) -> None:
    """Anonymously model initialization parameters for anonymized
    usage telemetry.

    At the moment, we only log the `fit_mode` parameter.

    Args:
        estimator: The TabPFN estimator instance.
        params: The model initialization parameters.
    """
    constructor = getattr(estimator.__class__, "__init__", None)
    if constructor is None:
        return

    # Create a validated copy of logged params; avoid passing in arbitrary params
    # as that would allow for PII leakage
    logged_params = {}

    signature_params = inspect.signature(constructor).parameters
    if "fit_mode" in params:
        param = signature_params.get("fit_mode")

        # Early return, may be replaced in the future when we start tracking
        # more parameters and validate their types.
        if not param:
            return

        annotation = param.annotation
        # Check if the annotation is a string and the fit_mode is in it.
        # An alternative may be evaluating the annotation, but this is more secure
        # because we don't want to execute arbitrary code.
        fit_mode = str(params["fit_mode"])
        if isinstance(param.annotation, str) and fit_mode in annotation:
            logged_params["fit_mode"] = fit_mode

    # Log the logged params
    if logged_params:
        try:
            # We conditionally import here to avoid introducing breaking changes as
            # this interface was introduced in tabpfn_common_utils 0.2.13 and not all
            # users have upgraded to this version yet.
            from tabpfn_common_utils.telemetry import set_init_params  # noqa: PLC0415

            set_init_params(logged_params)
        except ImportError:
            pass


def resolve_model_version(
    model_path: ModelPath | list[ModelPath] | None,
) -> ModelVersion:
    """Resolve the model version from the model path."""
    if isinstance(model_path, list):
        if len(model_path) == 0:
            return _resolve_model_version(None)
        resolved_model_versions = [_resolve_model_version(p) for p in model_path]
        if len(set(resolved_model_versions)) > 1:
            raise ValueError("All model paths must have the same version.")
        return resolved_model_versions[0]
    return _resolve_model_version(model_path)


def resolve_model_path(
    model_path: ModelPath | list[ModelPath] | None,
    which: Literal["regressor", "classifier"],
    version: Literal["v2", "v2.5"] = "v2.5",
) -> tuple[
    list[Path],
    list[Path],
    list[str],
    Literal["regressor", "classifier"],
]:
    """Resolves the model path, using the official default model if no path is provided.

    Args:
        model_path: An optional path to a model file. If None, the default
            model for the given `which` and `version` will be used, resolving
            to the local cache directory.
        which: The type of model ('regressor' or 'classifier').
        version: The model version (currently only 'v2').

    Returns:
        A tuple containing lists of resolved model Path(s),
        the parent directory Path(s), the model's filename(s), and model type Literal.
    """
    if model_path is None:
        # Get the source information to find the official default model filename.
        model_source = _get_model_source(ModelVersion(version), ModelType(which))
        resolved_model_names = [model_source.default_filename]

        # Determine the cache directory for storing models.
        resolved_model_dirs = [get_cache_dir()]
        resolved_model_paths = [resolved_model_dirs[0] / resolved_model_names[0]]
    elif isinstance(model_path, (str, Path)):
        resolved_model_paths = [Path(model_path)]
        resolved_model_dirs = [resolved_model_paths[0].parent]
        resolved_model_names = [resolved_model_paths[0].name]
    else:
        resolved_model_paths = [Path(p) for p in model_path]
        resolved_model_dirs = [p.parent for p in resolved_model_paths]
        resolved_model_names = [p.name for p in resolved_model_paths]

    return resolved_model_paths, resolved_model_dirs, resolved_model_names, which


def get_loss_criterion(
    config: ArchitectureConfig,
) -> nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution:
    """Create: for classification, a loss function. For regression, a BarDistribution.

    The classification loss is only required for training, but we always create it, for
    simplicity. The BarDistribution serves the dual purpose of loss function
    and output distribution, thus is required even during inference.
    """
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


def load_model(
    *,
    path: Path,
    cache_trainset_representation: bool = True,
) -> tuple[
    Architecture,
    nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution,
    ArchitectureConfig,
    InferenceConfig,
]:
    """Loads a model from a given path. Only for inference.

    Args:
        path: Path to the checkpoint
        cache_trainset_representation: If True, the model will cache the
            trainset representation. Forwarded to get_architecture.
    """
    # Catch the `FutureWarning` that torch raises. This should be dealt with!
    # The warning is raised due to `torch.load`, which advises against ckpt
    # files that contain non-tensor data.
    # This `weightes_only=None` is the default value. In the future this will default to
    # `True`, dissallowing loading of arbitrary objects.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        checkpoint: dict = torch.load(path, map_location="cpu", weights_only=None)

    try:
        architecture_name = checkpoint["architecture_name"]
    except KeyError:
        architecture_name = "base"
    architecture = ARCHITECTURES[architecture_name]
    state_dict = checkpoint["state_dict"]
    model_config, unused_model_config = architecture.parse_config(checkpoint["config"])
    logger.debug(
        "Keys in config that were not parsed by architecture config: "
        f"{', '.join(unused_model_config.keys())}"
    )

    criterion_state_keys = [k for k in state_dict if "criterion." in k]
    loss_criterion = get_loss_criterion(model_config)
    if isinstance(loss_criterion, FullSupportBarDistribution):
        # Remove from state dict
        criterion_state = {
            k.replace("criterion.", ""): state_dict.pop(k) for k in criterion_state_keys
        }
        loss_criterion.load_state_dict(criterion_state)
    else:
        assert len(criterion_state_keys) == 0, criterion_state_keys

    model = architecture.get_architecture(
        model_config,
        n_out=get_n_out(model_config, loss_criterion),
        cache_trainset_representation=cache_trainset_representation,
    )
    model.load_state_dict(state_dict)
    model.eval()

    inference_config = _get_inference_config_from_checkpoint(checkpoint, loss_criterion)

    return model, loss_criterion, model_config, inference_config


def _get_inference_config_from_checkpoint(
    checkpoint: dict[str, Any],
    criterion: nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution,
) -> InferenceConfig:
    """Return the config in the checkpoint, or an appropriate default config.

    We only started storing the inference config in the checkpoint after the v2.5
    release. Thus, if there is no config in the checkpoint, try to guess between v2 and
    v2.5 and get the correct config.
    """
    # This is how we tell the checkpoints apart:
    #     v2: "architecture_name" not present, as added after the v2 release
    #   v2.5: "architecture_name" present, but "inference_config" not present
    #  >v2.5: "inference_config" present, so don't need to guess a default config
    if inference_config := checkpoint.get("inference_config"):
        return InferenceConfig(**inference_config)
    if "architecture_name" not in checkpoint:
        model_version = ModelVersion.V2
    else:
        model_version = ModelVersion.V2_5

    if isinstance(criterion, FullSupportBarDistribution):
        task_type = "regression"
    else:
        task_type = "multiclass"

    return InferenceConfig.get_default(task_type, model_version)


def get_n_out(
    config: ArchitectureConfig,
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
    model: TabPFNRegressor | TabPFNClassifier,
    save_path: Path | str | list[Path | str],
    additional_fields: dict[str, Any] | None = None,
) -> None:
    """Save the underlying TabPFN foundation model to ``save_path``.

    This writes only the base pre-trained weights and configuration. It does
    **not** store a fitted :class:`TabPFNRegressor`/``Classifier`` instance.
    The resulting file is merely a checkpoint consumed by
    :func:`load_model_criterion_config` to build a new estimator.

    Args:
        model: The internal model object of a ``TabPFN`` estimator.
        save_path: Path to save the checkpoint to.
        additional_fields: Additional data to save to the checkpoint.

    """
    if len(model.models_) > 1 and (
        not isinstance(save_path, list) or len(save_path) != len(model.models_)
    ):
        raise ValueError(
            f"Your TabPFN estimator has multiple internal models ({len(model.models_)})"
            f", so you must provide a list of {len(model.models_)} save paths."
        )

    models = model.models_

    znorm_space_bardist = None
    if (
        hasattr(model, "znorm_space_bardist_")
        and model.znorm_space_bardist_ is not None  # type: ignore
    ):
        znorm_space_bardist = model.znorm_space_bardist_  # type: ignore

    configs = model.configs_
    save_paths = save_path if isinstance(save_path, list) else [save_path]

    for ens_model, config, path in zip(
        models,
        configs,
        save_paths,
    ):
        model_state = ens_model.state_dict()

        if znorm_space_bardist is not None:
            bardist_state = {
                f"criterion.{k}": v for k, v in znorm_space_bardist.state_dict().items()
            }
            state_dict = {**model_state, **bardist_state}
        else:
            state_dict = model_state

        checkpoint = {"state_dict": state_dict, "config": asdict(config)}

        if additional_fields is not None:
            checkpoint.update(additional_fields)

        torch.save(checkpoint, path)


def save_fitted_tabpfn_model(estimator: BaseEstimator, path: Path | str) -> None:
    """Persist a fitted TabPFN estimator to ``path``.

    This stores the initialization parameters and the fitted state, but crucially
    omits the large foundation model weights for efficiency.

    This does not support estimators using `fit_mode="fit_with_cache"`, and will raise
    NotImplementedError in this case.
    """
    if not hasattr(estimator, "executor_"):
        raise RuntimeError("Estimator must be fitted before saving.")

    path = Path(path)
    if path.suffix != ".tabpfn_fit":
        raise ValueError("Path must end with .tabpfn_fit")

    # Attributes that are handled separately or should not be saved.
    blacklist = {"models_", "executor_", "configs_", "devices_"}

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
        # move all tensors to "cpu" before saving, so if fitted & saved on cuda-device
        # and loading on cpu-device does not throw
        # "RuntimeError: Attempting to deserialize object on a CUDA device..."
        fitted_attrs = {
            k: v.to("cpu") if isinstance(v, (torch.nn.Module, torch.Tensor)) else v
            for k, v in fitted_attrs.items()
        }

        joblib.dump(fitted_attrs, tmp / "fitted_attrs.joblib")

        # 3. Save the InferenceEngine state without the model weights
        estimator.executor_.save_state_except_model_weights(
            tmp / "executor_state.joblib"
        )

        # 4. Create the final zip archive
        shutil.make_archive(str(path).replace(".tabpfn_fit", ""), "zip", tmp)
        shutil.move(str(path).replace(".tabpfn_fit", "") + ".zip", path)


def _extract_archive(path: Path, tmp: Path) -> None:
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
    path = Path(path)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        _extract_archive(path, tmp)

        # Load init params and create a fresh estimator instance
        with (tmp / "init_params.json").open() as f:
            params = json.load(f)

        saved_cls_name = params.pop("__class_name__")
        if isinstance(params.get("inference_precision"), str) and params[
            "inference_precision"
        ].startswith("torch."):
            dtype_name = params["inference_precision"].split(".")[1]
            params["inference_precision"] = getattr(torch, dtype_name)
        params["device"] = str(device)

        if saved_cls_name == "TabPFNClassifier":
            cls = import_module("tabpfn.classifier").TabPFNClassifier
        elif saved_cls_name == "TabPFNRegressor":
            cls = import_module("tabpfn.regressor").TabPFNRegressor
        else:
            raise TypeError(f"Unknown estimator class '{saved_cls_name}'")

        est = cls(**params)
        # This is critical: it loads the base model weights into `est.models_`
        est._initialize_model_variables()

        # Restore all other fitted attributes
        fitted_attrs = joblib.load(tmp / "fitted_attrs.joblib")
        for key, value in fitted_attrs.items():
            setattr(est, key, value)

        est.executor_ = InferenceEngine.load_state(
            tmp / "executor_state.joblib", est.models_
        )

        est.to(str(device))

        return est
