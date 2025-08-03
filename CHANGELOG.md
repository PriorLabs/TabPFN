# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
