"""Single-dataset fine-tuning wrappers for TabPFN models."""

from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier
from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor

__all__ = [
    "FinetunedTabPFNClassifier",
    "FinetunedTabPFNRegressor",
]
