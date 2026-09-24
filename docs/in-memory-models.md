# In-memory models

Pass `ModelSpecs` to `TabPFNClassifier` or `TabPFNRegressor` through `model_path`
to use an existing neural network without saving or loading a checkpoint:

```python
from tabpfn import ModelSpecs, TabPFNClassifier, TabPFNRegressor

# model, architecture_config, and inference_config come from your model setup.
specs = ModelSpecs(
    model=model,
    architecture_config=architecture_config,
    inference_config=inference_config,
)
classifier = TabPFNClassifier(model_path=specs)
regressor = TabPFNRegressor(model_path=specs)
```

The estimator selects the task. The same specs can back both estimators when the
network supports both tasks, as TabPFN-3.5 does. Wrapping a single-task model does
not add another task's head.

`ModelSpecs` is a dataclass, also available from `tabpfn.base`. It stores the model
and configuration objects by reference. It does **not** copy or freeze training
weights. For evaluation during training, create an isolated inference model first:
estimator use can change the model's device, precision, and inference state.

You can pass a list of specs to ensemble multiple models. Their inference settings
must satisfy the same compatibility checks as checkpoint ensembles, and their
regression distributions must have equal borders.

## Regression distributions and legacy models

`norm_criterion` is an optional `FullSupportBarDistribution` in **normalized target
space**. It interprets the model's regression logits as a continuous distribution,
including means and quantiles; it is not a choice of training objective.

When a regressor initializes:

1. An explicitly supplied `norm_criterion` takes precedence.
2. Otherwise, TabPFN constructs the distribution from `model.regression_borders`.
3. If neither is available, initialization raises an error requesting
   `norm_criterion`.

Models from v3 onward expose these borders, so modern in-memory training evaluation
needs no separate criterion construction or extraction. Classification ignores
`norm_criterion`.

Older inference models (v2/v2.5/v2.6) do not carry those borders. Retain the
regression distribution returned separately by `load_model`, or the estimator's
`znorm_space_bardist_` when finetuning or copying an estimator:

```python
from copy import deepcopy
from tabpfn import ModelSpecs, TabPFNRegressor

# trained_regressor has already initialized its model, e.g. through fit().
specs = ModelSpecs(
    model=deepcopy(trained_regressor.models_[0]),
    architecture_config=deepcopy(trained_regressor.configs_[0]),
    inference_config=deepcopy(trained_regressor.inference_config_),
    norm_criterion=deepcopy(trained_regressor.znorm_space_bardist_),
)
eval_regressor = TabPFNRegressor(model_path=specs)
```

A legacy training model may instead have `model.criterion`; pass that distribution
explicitly (copy it when isolating evaluation state). `ModelSpecs` does not inspect
that training attribute. Do not pass `raw_space_bardist_`: its borders have already
been transformed into the fitted dataset's target units.

## Migrating task-specific specs

`ClassifierModelSpecs`, `RegressorModelSpecs`, and `BaseModelSpecs` have been
removed from `tabpfn.base`. Replace their imports and constructors with
`ModelSpecs` (available from `tabpfn` or `tabpfn.base`). Its fields use the same
names, and the optional fourth positional argument is still `norm_criterion`.
Keep passing that distribution for regression models without embedded borders.

Previously `ModelSpecs` was a union type alias; it is now a concrete dataclass.
Use `isinstance(value, ModelSpecs)` to detect in-memory bundles. Code that
previously used `isinstance(value, RegressorModelSpecs)` to select a task must
instead use the requested estimator/task: a multitask bundle has no single task
identity. Serialized objects referencing the removed classes must be migrated
before upgrading or recreated with `ModelSpecs`.
