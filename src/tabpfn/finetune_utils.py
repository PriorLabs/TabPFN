"""Utilities for TabPFN model finetuning processes."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

# Importing submodules directly to avoid potential circular dependencies.
from tabpfn.classifier import TabPFNClassifier
from tabpfn.regressor import TabPFNRegressor
from tabpfn.base import ClassifierModelSpecs, RegressorModelSpecs

if TYPE_CHECKING:
    import numpy as np
    import torch

# TODO: temporary new file, move to
# Separate FineTuning folder soon

# TODO: passing eval_init_args is not optimal,
# since we are copying the model, we should
# be able to pass the original model to the
# evaluation model directly.


def clone_model_for_evaluation(
    original_model: TabPFNClassifier | TabPFNRegressor,
    eval_init_args: dict,
    model_class: type[TabPFNClassifier | TabPFNRegressor],
) -> TabPFNClassifier | TabPFNRegressor:
    """Prepares a deep copy of the model for
    evaluation to prevent modifying the original.
    Important in FineTuning since we are actively
    chaning the model being fine-tuned, however we
    still wish to evaluate it with our standard
    sklearn fit/predict inference interface.

    Args:
        original_model: The trained model instance
        (TabPFNClassifier or TabPFNRegressor).
        eval_init_args: Initialization arguments for
        the evaluation model instance.
        model_class: The class type (TabPFNClassifier
        or TabPFNRegressor) to instantiate.

    Returns:
        A new instance of the model class, ready for evaluation.
    """
    if hasattr(original_model, "model_") and original_model.model_ is not None:
        # Deep copy necessary components to avoid modifying the original trained model
        new_model_state = copy.deepcopy(original_model.model_)
        new_config = copy.deepcopy(original_model.config_)

        model_spec_obj = None
        if isinstance(original_model, TabPFNClassifier):
            model_spec_obj = ClassifierModelSpecs(
                model=new_model_state,
                config=new_config,
            )
        elif isinstance(original_model, TabPFNRegressor):
            # Regressor also needs the distribution criterion copied
            new_bar_dist = copy.deepcopy(original_model.bardist_)
            model_spec_obj = RegressorModelSpecs(
                model=new_model_state,
                config=new_config,
                norm_criterion=new_bar_dist,
            )
        else:
            raise TypeError("Unsupported model type for evaluation preparation.")

        eval_model = model_class(model_path=model_spec_obj, **eval_init_args)

    else:
        # If the original model hasn't been trained
        # or loaded, create a fresh one for eval
        eval_model = model_class(**eval_init_args)

    return eval_model


def create_evaluation_model(
    original_model: TabPFNClassifier | TabPFNRegressor,
    X_train: np.ndarray | torch.Tensor,
    y_train: np.ndarray | torch.Tensor,
    eval_init_args: dict,
    model_class: type[TabPFNClassifier | TabPFNRegressor],
) -> TabPFNClassifier | TabPFNRegressor:
    """Clone ``original_model`` and fit it for evaluation.

    This is a convenience wrapper around :func:`clone_model_for_evaluation`
    that immediately fits the cloned model on ``X_train`` and ``y_train`` so it
    can be used with :py:meth:`predict` or :py:meth:`predict_proba` in the
    standard inference mode.

    Parameters
    ----------
    original_model:
        The fine-tuned model instance to clone.
    X_train, y_train:
        Training data used to rebuild the inference engine for evaluation.
    eval_init_args:
        Initialization arguments for the evaluation model instance.
    model_class:
        The class to instantiate (``TabPFNClassifier`` or ``TabPFNRegressor``).

    Returns:
    -------
    TabPFNClassifier | TabPFNRegressor
        A fitted copy of ``original_model`` ready for evaluation.
    """
    eval_model = clone_model_for_evaluation(original_model, eval_init_args, model_class)
    eval_model.fit(X_train, y_train)
    return eval_model
