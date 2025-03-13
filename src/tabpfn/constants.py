"""Various constants used throughout the library."""

#  Copyright (c) Prior Labs GmbH 2025.

# TODO(eddiebergman): Should probably put these where they belong but
# for the time being, this just helps with typing and not the possible
# enumeration of things
from __future__ import annotations

from typing import Any, Literal
from typing_extensions import TypeAlias

import joblib
import numpy as np
from packaging import version

TaskType: TypeAlias = Literal["multiclass", "regression"]
TaskTypeValues: tuple[TaskType, ...] = ("multiclass", "regression")

# TODO
XType: TypeAlias = Any
SampleWeightType: TypeAlias = Any
YType: TypeAlias = Any
TODO_TYPE1: TypeAlias = str

NA_PLACEHOLDER = "__MISSING__"

SKLEARN_16_DECIMAL_PRECISION = 16
PROBABILITY_EPSILON_ROUND_ZERO = 1e-3
REGRESSION_NAN_BORDER_LIMIT_UPPER = 1e3
REGRESSION_NAN_BORDER_LIMIT_LOWER = -1e3
AUTOCAST_DTYPE_BYTE_SIZE = 2  # bfloat16
DEFAULT_DTYPE_BYTE_SIZE = 4  # float32

# Otherwise, yoa-johnson double power can end up causing a lot of overflows...
DEFAULT_NUMPY_PREPROCESSING_DTYPE = np.float64

# TODO(eddiebergman): Maybe make these a parameter
MEMORY_SAFETY_FACTOR = 5.0  # Taken as default from function


# TODO(eddiebergman): Pulled from `def get_ensemble_configurations()`
ENSEMBLE_CONFIGURATION_MAX_STEP = 2
MAXIMUM_FEATURE_SHIFT = 1_000
CLASS_SHUFFLE_OVERESTIMATE_FACTOR = 3

# 1) Figure out whether this Joblib version supports "generator_unordered".
# For example, assume "generator_unordered" is officially supported in joblib >= 1.4.0
SUPPORTS_GENERATOR_UNORDERED = version.parse(joblib.__version__) >= version.parse(
    "1.4.0",
)

# scikit-learn has a minimum of "1.2.0", thus we need to work also without return_as.
SUPPORTS_RETURN_AS = version.parse(joblib.__version__) >= version.parse(
    "1.3.0",
)
# 2) Define a mapping from your custom parallel mode to joblib's "return_as" parameter.
if SUPPORTS_GENERATOR_UNORDERED:
    # If the installed Joblib is new enough, allow "generator_unordered"
    PARALLEL_MODE_TO_RETURN_AS = {
        "block": "list",
        "in-order": "generator",
        "as-ready": "generator_unordered",
    }
else:
    # If the installed Joblib is older, fall back to "generator"
    PARALLEL_MODE_TO_RETURN_AS = {
        "block": "list",
        "in-order": "generator",
        # fallback to "generator" instead of "generator_unordered"
        "as-ready": "generator",
    }
