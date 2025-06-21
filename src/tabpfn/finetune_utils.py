"""Utilities for TabPFN model finetuning processes."""

from __future__ import annotations

import copy

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.base import ClassifierModelSpecs, RegressorModelSpecs

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
