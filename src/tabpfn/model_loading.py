"""Functions for downloading and loading model checkpoints."""

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
import warnings
import zipfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, overload
from urllib.error import URLError

import joblib
import torch
from torch import nn

from tabpfn.architectures import ARCHITECTURES
from tabpfn.architectures.base.bar_distribution import (
    BarDistribution,
    FullSupportBarDistribution,
)
from tabpfn.constants import ModelPath
from tabpfn.inference import InferenceEngine, InferenceEngineCacheKV
from tabpfn.settings import settings

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.architectures.interface import Architecture, ArchitectureConfig

logger = logging.getLogger(__name__)

# Public fallback base URL for model downloads
FALLBACK_S3_BASE_URL = "https://storage.googleapis.com/tabpfn-v2-model-files/05152025"


class ModelType(str, Enum):  # noqa: D101
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"


class ModelVersion(str, Enum):  # noqa: D101
    V2 = "v2"


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
        ]
        return cls(
            repo_id="Prior-Labs/TabPFN-v2-reg",
            default_filename="tabpfn-v2-regressor.ckpt",
            filenames=filenames,
        )


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
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
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
    model_path: ModelPath | list[ModelPath] | None,
    *,
    check_bar_distribution_criterion: Literal[False],
    cache_trainset_representation: bool,
    version: Literal["v2"],
    which: Literal["classifier"],
    download_if_not_exists: bool,
) -> tuple[
    list[Architecture],
    nn.BCEWithLogitsLoss | nn.CrossEntropyLoss,
    list[ArchitectureConfig],
]: ...


@overload
def load_model_criterion_config(
    model_path: ModelPath | list[ModelPath] | None,
    *,
    check_bar_distribution_criterion: Literal[True],
    cache_trainset_representation: bool,
    version: Literal["v2"],
    which: Literal["regressor"],
    download_if_not_exists: bool,
) -> tuple[
    list[Architecture], FullSupportBarDistribution, list[ArchitectureConfig]
]: ...


def load_model_criterion_config(
    model_path: ModelPath | list[ModelPath] | None,
    *,
    check_bar_distribution_criterion: bool,
    cache_trainset_representation: bool,
    which: Literal["regressor", "classifier"],
    version: Literal["v2"] = "v2",
    download_if_not_exists: bool,
) -> tuple[
    list[Architecture],
    nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution,
    list[ArchitectureConfig],
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
        download_if_not_exists: Whether to download the model if it doesn't exist.

    Returns:
        The model, criterion, and config.
    """
    (resolved_model_paths, resolved_model_dirs, resolved_model_names, which) = (
        _resolve_model_path(
            model_path=model_path,
            which=which,
            version=version,
        )
    )

    # TODO: Also make sure we don't load all models if only a single estimator
    # is used. Print a warning in this case.

    for folder in resolved_model_dirs:
        folder.mkdir(parents=True, exist_ok=True)

    loaded_models = []
    criterions = []
    configs = []

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
                version=version,
                which=cast("Literal['classifier', 'regressor']", which),
                model_name=resolved_model_names[i],
            )
            if res != "ok":
                repo_type = "clf" if which == "classifier" else "reg"
                raise RuntimeError(
                    f"Failed to download model to {path}!\n\n"
                    f"For offline usage, please download the model manually from:\n"
                    f"https://huggingface.co/Prior-Labs/TabPFN-v2-{repo_type}/resolve/main/{resolved_model_names[i]}\n\n"
                    f"Then place it at: {path}",
                ) from res[0]

        loaded_model, criterion, config = load_model(
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
        # TODO(ben): double-check if criterion match here
        criterions.append(criterion)
        configs.append(config)
    return loaded_models, criterions[0], configs


def _resolve_model_path(
    model_path: ModelPath | list[ModelPath] | None,
    which: Literal["regressor", "classifier"],
    version: Literal["v2"] = "v2",
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
        if settings.tabpfn.model_cache_dir is not None:
            resolved_model_dirs = [settings.tabpfn.model_cache_dir]
        else:
            resolved_model_dirs = [
                _user_cache_dir(platform=sys.platform, appname="tabpfn")
            ]
        resolved_model_paths = [resolved_model_dirs[0] / resolved_model_names[0]]
    elif isinstance(model_path, ModelPath):
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
        checkpoint = torch.load(path, map_location="cpu", weights_only=None)

    try:
        architecture_name = checkpoint["architecture_name"]
    except KeyError:
        architecture_name = "base"
    architecture = ARCHITECTURES[architecture_name]
    state_dict = checkpoint["state_dict"]
    config, unused_config = architecture.parse_config(checkpoint["config"])
    logger.debug(
        "Keys in config that were not parsed by architecture config: "
        f"{', '.join(unused_config.keys())}"
    )

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

    model = architecture.get_architecture(
        config,
        n_out=get_n_out(config, loss_criterion),
        cache_trainset_representation=cache_trainset_representation,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model, loss_criterion, config


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

        torch.save(checkpoint, path)


def save_fitted_tabpfn_model(
    estimator: BaseEstimator,
    path: Path | str,
) -> None:
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
        # This is critical: it loads the base model weights into `est.models_`
        est._initialize_model_variables()

        # 2. Restore all other fitted attributes
        fitted_attrs = joblib.load(tmp / "fitted_attrs.joblib")
        for key, value in fitted_attrs.items():
            setattr(est, key, value)

        # 3. Load the InferenceEngine state
        est.executor_ = InferenceEngine.load_state(tmp / "executor_state.joblib")

        # 4. Re-link the foundation model with the loaded engine
        if est.executor_.models is None:
            if isinstance(est.executor_, InferenceEngineCacheKV):  # type: ignore
                est.executor_.models = [
                    deepcopy(est.models_[config._model_index])
                    for config in est.executor_.ensemble_configs  # type: ignore
                ]
            else:
                est.executor_.models = [deepcopy(model) for model in est.models_]

        # 5. Move all torch components to the target device
        est.devices_ = (torch.device(device),)
        if hasattr(est.executor_, "models"):
            est.executor_.models = [m.to(device) for m in est.executor_.models]

        # Restore other potential torch objects from fitted_attrs
        for key, value in vars(est).items():
            if key.endswith("_") and hasattr(value, "to"):
                setattr(est, key, value.to(device))

        return est
