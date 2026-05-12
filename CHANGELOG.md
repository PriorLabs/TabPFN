# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [8.0.1] - 2026-05-12

### Fixed

- Remove warning about SVD falling back to CPU on MPS. ([#957](https://github.com/PriorLabs/TabPFN/pull/957))


## [8.0.0] - 2026-05-12

### Breaking Changes

- **Major release**: TabPFN-3 is now the default model. New users and existing users who do not pin a model will automatically get TabPFN-3 going forward. To use a previous model version, use the `create_default_for_version()` classmethod on `TabPFNClassifier` / `TabPFNRegressor`, or pass an explicit `model_path` to the estimator constructor to pin a specific model file. ([#948](https://github.com/PriorLabs/TabPFN/pull/948))

### Added

- Add opt-in feature subsampling strategies across ensemble members when the number of features exceeds `max_features_per_estimator`. Set `FEATURE_SUBSAMPLING_METHOD` in the inference config to one of `"random"` (default), `"balanced"`, or `"constant_and_balanced"`. ([#851](https://github.com/PriorLabs/TabPFN/pull/851))
- Add enable_torch_compile to PerformanceOptions. ([#879](https://github.com/PriorLabs/TabPFN/pull/879))
- Add GPU preprocessing pipeline that runs feature transformations (quantile normalization, SVD) directly on the GPU as part of the model forward pass. ([#884](https://github.com/PriorLabs/TabPFN/pull/884))
- Add `get_inference_config()` method to `TabPFNClassifier` and `TabPFNRegressor`. This method loads the model checkpoint if needed and returns the active `InferenceConfig`, allowing inspection of preprocessing and inference settings before calling `fit()`. ([#890](https://github.com/PriorLabs/TabPFN/pull/890))
- Add an optional `show_progress_bar` flag to TabPFN classifier and regressor inference, defaulting to `False`. ([#899](https://github.com/PriorLabs/TabPFN/pull/899))
- Add a nightly workflow that reproduces every example notebook's pip-install sequence in a fresh venv and asserts `tabpfn` resolves to the latest PyPI release. ([#901](https://github.com/PriorLabs/TabPFN/pull/901))
- Add `gini_feature_importance` and `gini_feature_importance_lightgbm` as new `FEATURE_SUBSAMPLING_METHOD` options. Both rank features by importance and always include the top-K most predictive features per estimator when the dataset exceeds `max_features_per_estimator`. LightGBM is an optional dependency (`pip install tabpfn[lightgbm]`). ([#908](https://github.com/PriorLabs/TabPFN/pull/908))
- Add TabPFN v3 support: `TabPFNClassifier` and `TabPFNRegressor` now support `ModelVersion.V3`, including `create_default_for_version(ModelVersion.V3)` and explicit v3 model paths. ([#909](https://github.com/PriorLabs/TabPFN/pull/909))
- Add `auto` as a new `FEATURE_SUBSAMPLING_METHOD` option. When selected, it automatically uses `gini_feature_importance` (LightGBM-based) for datasets with more than 100k samples where feature subsampling is needed, and falls back to `balanced` otherwise. LightGBM is now a required dependency (previously optional via `pip install tabpfn[lightgbm]`). ([#913](https://github.com/PriorLabs/TabPFN/pull/913))
- Add `embedding_dim` abstract property to the `Architecture` interface, exposing the output embedding dimension for all architecture implementations. ([#924](https://github.com/PriorLabs/TabPFN/pull/924))
- Stratified row subsampling for the classifier: when `SUBSAMPLE_SAMPLES` is set, each ensemble member now draws rows that preserve the original class proportions, using a balanced round-robin pool per class to ensure uniform row coverage across estimators. ([#928](https://github.com/PriorLabs/TabPFN/pull/928))
- Add opt-in FlashAttention-3 backend selector for v3 (`PerformanceOptions.attention_backend`). On Hopper GPUs, "auto" routes to FA3 once the sequence length amortises FA3's dispatch overhead; otherwise falls back to PyTorch SDPA. ([#935](https://github.com/PriorLabs/TabPFN/pull/935))
- Auto-scale `n_estimators` at fit time so every feature is covered by at least one ensemble member. The effective count is exposed as `n_estimators_`; a `UserWarning` is emitted when scaling triggers. ([#937](https://github.com/PriorLabs/TabPFN/pull/937))
- Add `TorchSquashingScaler` and `TorchSquashingScalerStep` — a torch implementation of `SquashingScaler` mirroring the CPU version. ([#938](https://github.com/PriorLabs/TabPFN/pull/938))
- Run SVD on GPU when `enable_gpu_preprocessing=True` by pre-warming PyTorch's LAPACK lazy wrapper on the main thread before parallel dispatch to avoid a multi-GPU race in `torch.svd_lowrank` -> `torch.linalg.qr`. ([#941](https://github.com/PriorLabs/TabPFN/pull/941))
- Schedule the squashing scaler on GPU when the configuration is eligible. This makes the preprocessing significantly faster. ([#944](https://github.com/PriorLabs/TabPFN/pull/944))

### Changed

- Introduces balanced subsampling of features for improved performance for datasets with large number of features. Results may vary slightly because of different seeds. ([#851](https://github.com/PriorLabs/TabPFN/pull/851))
- Model checkpoint caching now automatically invalidates when the file on disk changes (detected via mtime and size), so replaced checkpoints (e.g. during finetuning) are always reloaded. ([#863](https://github.com/PriorLabs/TabPFN/pull/863))
- Row subsampling across ensemble members now uses round-robin balanced sampling. This replaces the previous random sampling approach. ([#886](https://github.com/PriorLabs/TabPFN/pull/886))
- Remove unused v2.6 defaults from `InferenceConfig.get_default()`. V2.6 checkpoints always embed their own `InferenceConfig`, so these defaults were never used at inference time. The v2.6 preprocessor config factories are also removed from `tabpfn.preprocessing`. ([#890](https://github.com/PriorLabs/TabPFN/pull/890))
- Renamed `InferenceConfig.CONSTANT_FEATURE_COUNT` to `FEATURE_SUBSAMPLING_CONSTANT_FEATURE_COUNT` to better reflect its purpose. Old checkpoints that store the previous key name are migrated transparently on load. ([#900](https://github.com/PriorLabs/TabPFN/pull/900))
- Updated copyright year to 2026 and consolidated the `authors` field in `pyproject.toml` to a single Prior Labs entry. ([#916](https://github.com/PriorLabs/TabPFN/pull/916))
- Speed up `ReshapeFeatureDistributionsStep` ~2x on large numerical workloads (~1670 ms → ~870 ms on 100k×100): inline `SquashingScaler`'s robust/minmax branches into a single `nanpercentile` pass, and call `ColumnTransformer.fit_transform` once instead of `fit` + `transform` (sklearn's `fit` already runs the transform internally). Behavior unchanged. ([#938](https://github.com/PriorLabs/TabPFN/pull/938))
- Keep the inference cache on the GPU by default when `fit_mode="fit_with_cache"`, avoiding host/device transfers on each predict call. The per-estimator KV caches are reachable via `model.executor_.kv_caches`. ([#942](https://github.com/PriorLabs/TabPFN/pull/942))
- Clean up README and inline references to removed/deprecated tabpfn-extensions modules (`rf_pfn`, `post_hoc_ensembles`, `hpo`) and the retired `large_datasets` example. Drops the now-stale workflow mermaid diagram, updates the OOM error message to link to the Models page, and removes the unused `AutoTabPFNClassifier` import from the Colab demo notebook. ([#945](https://github.com/PriorLabs/TabPFN/pull/945))

### Fixed

- Fix inference precision to respect force_inference_dtype in KV cache engine and skip thinking tokens during cache-building. ([#802](https://github.com/PriorLabs/TabPFN/pull/802))
- Reduce TabPFNRegressor peak GPU memory at large test-set sizes by chunking the row dimension inside `translate_probs_across_borders`. Output is unchanged; peak drops ~60% at `n_test=250k` (57.6 GB → 22.8 GB on an H100). ([#882](https://github.com/PriorLabs/TabPFN/pull/882))
- Fix v2.6 producing near-random outputs on Apple Silicon (MPS). `F.scaled_dot_product_attention` on MPS silently returns wrong values for non-contiguous q/k/v (upstream: pytorch/pytorch#181133); we now force contiguity before the call. Iris multiclass accuracy on MPS: 0.48 → 0.98. ([#888](https://github.com/PriorLabs/TabPFN/pull/888))
- Fix `FinetunedTabPFNClassifier`/`FinetunedTabPFNRegressor` dropping pandas feature names from the final inference model. The raw training inputs are now retained so the fitted inference estimator records `feature_names_in_`, and calling `predict_proba`/`predict` with a DataFrame no longer triggers spurious sklearn feature-name warnings. ([#892](https://github.com/PriorLabs/TabPFN/pull/892))
- Adapt `recompute_layer` flag in `FinetunedTabPFNClassifier`/`FinetunedTabPFNRegressor` to new `PerformanceOptions` interface. ([#917](https://github.com/PriorLabs/TabPFN/pull/917))
- Fix `save_tabpfn_model` not setting `architecture_name="tabpfn_v3"` for v3 configs and not persisting `inference_config_`, which broke resuming v3 finetuning from a saved checkpoint. ([#930](https://github.com/PriorLabs/TabPFN/pull/930))
- Reduce KV cache GPU memory in `fit_with_cache` by materialising only the kept KV head(s) at cache-build time. Output is unchanged. ([#933](https://github.com/PriorLabs/TabPFN/pull/933))
- Fix `RuntimeError: No available kernel` on v3 inference for GPUs where none of FlashAttention / EfficientAttention / CuDNN-Attention are eligible (e.g. Turing-class cards like the T4) by adding `SDPBackend.MATH` as a final fallback in `_SDPA_BACKENDS`. ([#947](https://github.com/PriorLabs/TabPFN/pull/947))


## [7.1.1] - 2026-04-09

### Added

- Add modular experiment logging for finetuning with `experiment_logger` parameter, including `WandbLogger` for W&B tracking and a `FinetuningLogger` protocol for custom integrations. ([#815](https://github.com/PriorLabs/TabPFN/pull/815))
- Add three-tier authentication flow: browser-based login for graphical environments, headless interactive login with clipboard copy for SSH/cluster sessions, and clear step-by-step instructions for fully non-interactive environments. ([#862](https://github.com/PriorLabs/TabPFN/pull/862))

### Changed

- - Optimize regressor predict method for memory efficiency
    - Average ensemble outputs on-the-fly instead of accumulating all outputs
    - Reduces memory usage by avoiding storage of all intermediate outputs, especially beneficial for large `n_estimators` ([#745](https://github.com/PriorLabs/TabPFN/pull/745))

### Fixed

- Fix bugs where fit_mode="fit_with_cache" produced slightly incorrect predictions in v2.5 (but not v2): thinking tokens were added twice, `inference_precision` flag was not applied correctly. ([#852](https://github.com/PriorLabs/TabPFN/pull/852))


## [7.1.0] - 2026-04-02

### Added

- More informative Out-Of-Memory error message. ([#805](https://github.com/PriorLabs/TabPFN/pull/805))
- Add multi-GPU DDP support for finetuning via torchrun (auto-detected, no code changes needed) ([#812](https://github.com/PriorLabs/TabPFN/pull/812))
- Add task_type to forward. ([#844](https://github.com/PriorLabs/TabPFN/pull/844))
- Exclude very recent package release in environment ([#847](https://github.com/PriorLabs/TabPFN/pull/847))

### Changed

- Switch from Hugging Face to Prior Labs website for model license acceptance ([#798](https://github.com/PriorLabs/TabPFN/pull/798))
- "auto" device selection now uses all available CUDA GPUs instead of only the first one ([#808](https://github.com/PriorLabs/TabPFN/pull/808))
- Optimize fingerprint hashing in preprocessing: round feature matrix once instead of per-row, avoid redundant SHA-256 calls. Speeds up fit by up to 2x for large datasets. ([#818](https://github.com/PriorLabs/TabPFN/pull/818))

### Fixed

- Fix the pdf() in FullSupportBarDistribution to actually compute the probability density. ([#799](https://github.com/PriorLabs/TabPFN/pull/799))
- Fix float overflow in Yeo-Johnson inverse transform that produced `inf` values and silently degraded regression border resolution. ([#838](https://github.com/PriorLabs/TabPFN/pull/838))
- Fix differentiable input for v2.6 ([#843](https://github.com/PriorLabs/TabPFN/pull/843))


## [7.0.1] - 2026-03-26

### Added

- Remove the n_out parameter from get_architecture. ([#839](https://github.com/PriorLabs/TabPFN/pull/839))

### Changed

- Make TabPFN-2.6 the default model ([#840](https://github.com/PriorLabs/TabPFN/pull/840))


## [7.0.0] - 2026-03-24

### Added

- Introduce TabPFN-2.6 model and use as default ([#831](https://github.com/PriorLabs/TabPFN/pull/831))
- Added argument `use_fixed_preprocessing_seed` to `FinetunedTabPFNClassifier` and `FinetunedTabPFNRegressor` for improved finetuning performance.
- This PR changes the random seeds used in the preprocessing, which may cause slight differences in final outcomes compared to previous versions.
  ([#771](https://github.com/PriorLabs/TabPFN/pull/771))
- More informative Out-Of-Memory error message. ([#805](https://github.com/PriorLabs/TabPFN/pull/805))
- Added `max_onehot_cardinality` option to cap one-hot encoding expansion for high-cardinality categorical features. ([#833](https://github.com/PriorLabs/TabPFN/pull/833))

### Changed

- Introduces TabPFN-2.6 as the new default model for TabPFNClassifier and TabPFNRegressor ([#831](https://github.com/PriorLabs/TabPFN/pull/831))
- Remove unused functions `default_classifier_preprocessor_configs()` and `default_regressor_preprocessor_configs()` ([#831](https://github.com/PriorLabs/TabPFN/pull/831))
- "auto" device selection now uses all available CUDA GPUs instead of only the first one ([#808](https://github.com/PriorLabs/TabPFN/pull/808))
- Optimize fingerprint hashing in preprocessing: round feature matrix once instead of per-row, avoid redundant SHA-256 calls. Speeds up fit by up to 2x for large datasets. ([#818](https://github.com/PriorLabs/TabPFN/pull/818))
- Bump minimum torch version from 2.1 to 2.5 ([#823](https://github.com/PriorLabs/TabPFN/pull/823))
- Cache loaded checkpoints across fit calls: skip redundant disk I/O when the same model is loaded repeatedly (e.g. cross-validation, hyperparameter search). ([#832](https://github.com/PriorLabs/TabPFN/pull/832))

### Fixed

- Fix the pdf() in FullSupportBarDistribution to actually compute the probability density. ([#799](https://github.com/PriorLabs/TabPFN/pull/799))


## [6.4.1] - 2026-02-19

### Changed

- Download lock is now scoped to the target file path, allowing concurrent downloads of different model files to proceed in parallel instead of serializing all downloads behind a single global lock. ([#790](https://github.com/PriorLabs/TabPFN/pull/790))


## [6.4.0] - 2026-02-18

### Added

- Introduces dedicated method for fitting with differentiable input called `fit_with_differentiable_input()` ([#752](https://github.com/PriorLabs/TabPFN/pull/752))
- Pass through kwargs in FinetunedTabPFNClassifier and FinetunedTabPFNRegressor predict and predict_proba methods to allow additional options like output_type='full' ([#772](https://github.com/PriorLabs/TabPFN/pull/772))
- Add MPS memory limiting to prevent macOS system crashes when using Apple Silicon GPUs. Memory is automatically limited to 70% of recommended max on import. Configurable via `TABPFN_MPS_MEMORY_FRACTION` environment variable. ([#773](https://github.com/PriorLabs/TabPFN/pull/773))
- Added `TabPFNCUDAOutOfMemoryError` and `TabPFNMPSOutOfMemoryError` for GPU out-of-memory errors during prediction with large test sets, providing helpful guidance on batching predictions. ([#774](https://github.com/PriorLabs/TabPFN/pull/774))

### Changed

- Remove upper version limits on dependencies ([#764](https://github.com/PriorLabs/TabPFN/pull/764))
- Refactored preprocessing pipeline:
  * Introduced `FeatureSchema` system to track column metadata through transformations, replacing raw categorical index lists.
  * Added `PreprocessingPipeline` and `PreprocessingStep` interfaces for modular transformations and updated all preprocessing steps.
  * Added `TabPFNLabelEncoder` for centralized label validation and metadata extraction.

  ([#767](https://github.com/PriorLabs/TabPFN/pull/767))
- * Introduces AddSVDFeaturesStep as a dedicated preprocessing step for SVD feature generation
  * Removes SVD-related functionality from ReshapeFeatureDistributionsStep
  * Extracts utility functions to a new `tabpfn/preprocessing/steps/utils.py` module

  ([#768](https://github.com/PriorLabs/TabPFN/pull/768))
- SVD preprocessing is now applied after categorical encoding for more robustness. Note that this may result in slight variations in final outcomes compared to previous versions. ([#779](https://github.com/PriorLabs/TabPFN/pull/779))
- Remove `random_state` parameter from `AddFingerprintFeaturesStep`; fingerprint hashing is now fully deterministic and no longer uses a random salt. Predictions will differ slightly from previous versions due to the changed fingerprint values. ([#780](https://github.com/PriorLabs/TabPFN/pull/780))
- Fix bug related to column ordering in ordinal encoder by introducing `OrderPreservingColumnTransformer`. Note that this change can cause slight differences in final outcomes compared to previous versions. ([#788](https://github.com/PriorLabs/TabPFN/pull/788))

### Fixed

- Fix race condition when model is downloaded simultaneously by multiple processes ([#738](https://github.com/PriorLabs/TabPFN/pull/738))
- Fix infinite loop in fingerprint hashing when rows contain inf or very large floats ([#780](https://github.com/PriorLabs/TabPFN/pull/780))

### Deprecated

- Removes "scaler" as an option for `global_transformer_name` in `PreprocessorConfig` ([#768](https://github.com/PriorLabs/TabPFN/pull/768))


## [6.3.2] - 2026-01-30

### Added

- - Moved preprocessing-related code to dedicated modules inside `src/tabpfn/preprocessing/`
  - Renamed public functions: 
      - `validate_X_predict` → `ensure_compatible_predict_input_sklearn`
      - `validate_Xy_fit` → `ensure_compatible_fit_inputs_sklearn`

  ([#720](https://github.com/PriorLabs/TabPFN/pull/720))
- - Add new features to finetuning (metric selection, time limit, passing validation data)
    - Added `eval_metric` and `time_limit` parameters to `FinetunedTabPFNClassifier` and `FinetunedTabPFNRegressor` 
    - Added `X_val`, `y_val` parameters to `.fit()` of `FinetunedTabPFNClassifier` and `FinetunedTabPFNRegressor` 
  - Fix bug in finetuning for splitting very small datasets
  - Ensure finetuning compares to the default checkpoint and does not accept worse models after finetuning

  ([#730](https://github.com/PriorLabs/TabPFN/pull/730))
- - Ensure `TabPFNValidationError` wraps both custom and sklearn's validate_data() errors ([#732](https://github.com/PriorLabs/TabPFN/pull/732))
- Refactor of model encoder. Move imports from `tabpfn.architectures.base.encoders` to `tabpfn.architectures.encoders` ([#733](https://github.com/PriorLabs/TabPFN/pull/733))
- Renamed the estimator's `preprocessor_` attribute to `ordinal_encoder_` ([#756](https://github.com/PriorLabs/TabPFN/pull/756))
- Pass through kwargs in `FinetunedTabPFNClassifier` and `FinetunedTabPFNRegressor` predict and predict_proba methods to allow additional options like `output_type='full'` ([#772](https://github.com/PriorLabs/TabPFN/pull/772)) 


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
