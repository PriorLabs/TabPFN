# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [6.3.1] - 2026-01-14

### Added

- Ensure `TabPFNValidationError` wraps both custom and sklearn's validate_data() errors

## [6.3.0] - 2026-01-06

### Added

- Fix sklearn issue making new tests fail by @noahho in https://github.com/PriorLabs/TabPFN/pull/698
- Fix KDI transformer init signature for sklearn compatibility by @noahho in https://github.com/PriorLabs/TabPFN/pull/696
- Improved analytics for tracking usage of different fit modes by @safaricd in https://github.com/PriorLabs/TabPFN/pull/646
- Add finetuning wrapper for classifier by @bejaeger in https://github.com/PriorLabs/TabPFN/pull/701
- Add Enterprise Edition section to README by @noahho in https://github.com/PriorLabs/TabPFN/pull/704
- [WIP] Refactor preprocessing into preprocessors package by @noahho in https://github.com/PriorLabs/TabPFN/pull/697
- Make fitted attributes safe by @noahho in https://github.com/PriorLabs/TabPFN/pull/707
- Document available checkpoints on Hugging Face by @LeoGrin in https://github.com/PriorLabs/TabPFN/pull/690
- Custom error for input validation by @simo-prior in https://github.com/PriorLabs/TabPFN/pull/692

## [6.2.0] - 2025-12-18

### Added
- Add a `.to()` method to `TabPFNClassifier` and `TabPFNRegressor`, allowing the device to be changed after `.fit()` has been called. This change also stores the model on the GPU between `.fit()` and `.predict()` calls, use `.to("cpu")` to release this GPU memory. [#685](https://github.com/PriorLabs/TabPFN/pull/685)

### Changed

## [6.1.0] - 2025-12-15

### Added

- Allow `SUBSAMPLE_SAMPLES` in `InferenceConfig` to take a list of list of indices to subsample for each estimator [#622](https://github.com/PriorLabs/TabPFN/pull/622)

### Changed

- Don't select MPS devices below PyTorch 2.5 and raise an error if selected, due to poor performance [#619](https://github.com/PriorLabs/TabPFN/pull/619)
- In multi-GPU inference, cache the model(s) on each device between estimators, to improve speed [#628](https://github.com/PriorLabs/TabPFN/pull/628)
- Fix crash if model is loaded and then saved again [#672](https://github.com/PriorLabs/TabPFN/pull/672)

## [6.0.6] - 2025-11-10

### Added
- Add a link to the gated model docs to the error message [#613](https://github.com/PriorLabs/TabPFN/pull/613)
- Anonymously report on used `model_path` and `model_version` [#611](https://github.com/PriorLabs/TabPFN/pull/611)

## [6.0.1] - 2025-11-06

### Changed

- Updated automatic selection of memory saving mode to improve fit + predict speed [#605](https://github.com/PriorLabs/TabPFN/pull/605)

## [6.0.0] - 2025-11-06

### Added

- Released TabPFN-2.5, a strong improvement over TabPFNv2 scaling to datasets with up to 50,000 samples and 2,000 features (more details [here](https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report)). This is used by default when using package version 6.0.0 and higher. To use the previous version, use `from tabpfn.constants import ModelVersion; TabPFNClassifier.create_default_for_version(ModelVersion.V2)`. Note that TabPFN-2.5 is released under a new [TABPFN-2.5 Non-Commercial License v1.0 license](https://huggingface.co/Prior-Labs/tabpfn_2_5/blob/main/LICENSE).

### Changed

- Deprecated the parameters `TabPFNClassifier(n_jobs=...)` and
  `TabPFNRegressor(n_jobs=...)` which had no effect, and replaced them with
  functioning `n_preprocessing_jobs`. We strongly recommend using the default value of
  `1`. [#555](https://github.com/PriorLabs/TabPFN/pull/555)
- Introduced interface to use `TabPFNClassifier` and `TabPFNRegressor` with multiple models in an ensemble. [#557](https://github.com/PriorLabs/TabPFN/pull/557)
- Fix precision of model outputs in the case when `softmax_temperature=1.0` [#569](https://github.com/PriorLabs/TabPFN/pull/569)
- Rename `tabpfn.config.ModelInterfaceConfig` to `tabpfn.inference_config.InferenceConfig` [#575](https://github.com/PriorLabs/TabPFN/pull/575)
- Add option to `TabPFNClassifier` to calibrate probabilities and tune decision thresholds for a specified metric. The feature can be used by specifying `eval_metric` and `tuning_config` during initialization [#218](https://github.com/PriorLabs/TabPFN-private/pull/218)
- Change `ensure_y_numeric=False` for `TabPFNRegressor` to `True` - need to validate `y_train` contains numerics.

## [2.2.1] - 2025-09-17

### Changed

- Fixed bug on multi-GPU systems leading to worse results

## [2.2.0] - 2025-09-15

### Added

### Changed

- Refactored preprocessing-related code [#503](https://github.com/PriorLabs/TabPFN/pull/503).
- Improved speed of `QuantileTransformer` for sample sizes larger 10k. This change also leads to subtle changes (improving the outcomes of the transformer slightly) at large sample sizes. [#503](https://github.com/PriorLabs/TabPFN/pull/503).
- @safaricd Clarified details of anonymous usage telemetry collection.

### Bug Fixes

## [2.1.4] - 2025-09-11 - **yanked**

### Added

### Changed

- @benraha Improved the inference speed on CPU significantly [#459](https://github.com/PriorLabs/TabPFN/pull/459).
- @benraha Added a fast-path for the column selection in RemoveEmptyFeaturesEncoderStep [#468](https://github.com/PriorLabs/TabPFN/pull/468).
- @safaricd Added anonymous usage analytics [#499](https://github.com/PriorLabs/TabPFN/pull/499)
- `TabPFNClassifier/Regressor.device_` has been replaced with `.devices_` [#496](https://github.com/PriorLabs/TabPFN/pull/496).

### Bug Fixes

## [2.1.3] - 2025-08-26

### Added

- Added several new finetuned model checkpoints. ([#462](https://github.com/PriorLabs/TabPFN/pull/462))

### Changed

### Bug Fixes

- Current infer categoricals crashes in case user tries to pass a feature as input that contains str and nan values. ([#432](https://github.com/PriorLabs/TabPFN/pull/432))
- Fixed a validation error that occurred when a `.env` file contained settings from other applications. ([#446](https://github.com/PriorLabs/TabPFN/pull/446))
- Fixed a crash on PyTorch versions older than 2.5 by correctly detecting Grouped-Query Attention (GQA) support. ([#438](https://github.com/PriorLabs/TabPFN/pull/438))

## [2.1.2] - 2025-08-03

- No changes -

## [2.1.1] - 2025-08-03

### Added

- Added a new `predict_logits()` method to `TabPFNClassifier` to return raw model outputs (logits). This is useful for model explainability tasks (e.g., with SHAP) that benefit from unnormalized, additive outputs.
- Support for MPS device: TabPFN can run on local Apple MPS Accelerator.

### Changed

- Increased the default value of the `n_estimators` parameter in `TabPFNClassifier` from `4` to `8`. This change aims to improve average accuracy by default, with the trade-off of increased inference time and memory usage. ([#384](https://github.com/PriorLabs/TabPFN/pull/384))
- Refactored the internal prediction logic for `TabPFNClassifier` for improved clarity, modularity, and maintainability.
- Regression finetuning outputs are renamed to more clearly reflect their purpose.
- Updated the Colab Notebook to include more of TabPFNs functionality (Row embeddings, string input data, missing value imputation, time series forecasting).
- Classifier finetunging now operates on the logits directly.

### Bug fix

- @benraha fixed a bug with differentiable inputs to the TabPFNClassifer.
- @zhengaq fixed a bug when a row was completely consisting of missing values.
- @rosenyu304 fixed a bug with the random number generator for old sklearn versions.

## [2.1.0] - 2025-07-04

### Changed

- **New Default Model**: The default classifier model has been updated to a new finetuned version (`tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`) to improve out-of-the-box performance.
- **Overhauled Examples**: The finetuning examples (`finetune_classifier.py`, `finetune_regressor.py`) have been completely rewritten with a clearer structure, centralized configuration, and more robust evaluation.
- Simplified `ignore_pretraining_limits` behavior by removing redundant warnings when the flag is enabled.

### Fixed

- The model now automatically switches between `fit_mode='batched'` and standard modes when calling `fit()` and `fit_from_preprocessed()`. This prevents crashes and provides a smoother finetuning experience by logging a warning instead of raising an error.
