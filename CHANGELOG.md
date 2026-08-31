# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The softmax temperature is now part of the inference config (`InferenceConfig.SOFTMAX_TEMPERATURE`), so a checkpoint can ship the temperature it was trained for instead of the interface hard-coding one for every model. The `softmax_temperature` argument of `TabPFNClassifier`/`TabPFNRegressor` now defaults to `"auto"`, meaning "use what the checkpoint declares"; passing a float still overrides it for every model, as before. Behavior is unchanged for existing checkpoints: none of them carry the new field, so they fall back to the field's default of `0.9`, which is the value the interface used to apply. Models in a multi-checkpoint ensemble may now declare different temperatures -- each estimator is scaled by the temperature of the model it ran on, exposed as `softmax_temperatures_` -- where before, checkpoints with differing inference configs were rejected outright.

## [8.5.0] - 2026-08-27

### Breaking Changes

- Cache the decoder keys instead of the train embeddings. ` get_embeddings(model, X_test, data_source="train")` is not supported with cached infernce anymore. ([#1189](https://github.com/PriorLabs/TabPFN/pull/1189))
- `tabpfn.model_loading.download_all_models()` now raises an exception if one or more of the models fails to download. It will still download all possible models before raising the exception. ([#1195](https://github.com/PriorLabs/TabPFN/pull/1195))

### Added

- `fit()` now recognizes a date-like string column internally, though nothing yet expands it into calendar features: it is still read as a plain category or text. We also added `InferenceConfig.MIN_CARDINALITY_FOR_TEXT`, to differentiate between category-vs-text and category-vs-number decisions; they default to the same value, since we are still not handling text. ([#1205](https://github.com/PriorLabs/TabPFN/pull/1205))

### Changed

- Reduced peak host memory during preprocessing: the ensemble preprocessor no longer rebuilds the feature matrix in steps that cannot change it, taking transient RSS from 42.7 GB to 12.0 GB (-72%), and wall time with it, on a 666,667 x 2,000 float64 fit. Preprocessed outputs are unchanged. ([#1186](https://github.com/PriorLabs/TabPFN/pull/1186))
- Reduced peak host memory during preprocessing for tables with categorical columns: the reshape and ordinal-encoding steps no longer rebuild the feature matrix to reorder it or to encode part of it, taking transient RSS from 3.33 GB to 2.80 GB (-16%) on a 333,333 x 400 half-categorical fit. Preprocessed outputs are unchanged. ([#1187](https://github.com/PriorLabs/TabPFN/pull/1187))
- Speed up modality detection on large string columns. Deciding whether a column holds numbers or dates now stops at the first value that does not parse within a 1024-row prefix, instead of parsing every row first. Detection of a 1-million-row free-text column drops from roughly 14 seconds to under 20 milliseconds; the answers are unchanged. ([#1208](https://github.com/PriorLabs/TabPFN/pull/1208))

### Fixed

- Fix an "illegal memory access" crash in the backward pass when fine-tuning on large batches: FlashAttention's backward indexes its workspace with 32-bit integers, so the batch is now chunked to keep each call inside that range. ([#1184](https://github.com/PriorLabs/TabPFN/pull/1184))
- Fix `fit_mode="fit_with_cache"` raising `TypeError: forward() missing 1 required positional argument: 'task_type'` for architectures whose forward pass takes a `task_type`: the KV cache build now forwards it, like the prediction paths already did. ([#1197](https://github.com/PriorLabs/TabPFN/pull/1197))
- `fit()` no longer crashes the interpreter outright on a table whose text column holds a hash-like value such as `"8e2569614270f3d8b9e7038efac9f116"`. Modality detection asked `pandas.to_numeric` whether a column was numeric; below pandas 3.0 that function has a signed 32-bit integer overflow in its scientific-notation parser and segfaults on a string whose exponent lands in `[2**31, 2**32)` ([pandas#63650](https://github.com/pandas-dev/pandas/issues/63650), fixed upstream in pandas 3.0). A segfault cannot be caught with `try`/`except`, so below pandas 3.0 the check now reads one value at a time with Python's built-in `float`, which does not share the bug. On pandas 3.0 and later the check is unchanged. ([#1203](https://github.com/PriorLabs/TabPFN/pull/1203))
- Fix `TabPFNRegressor` rejecting a checkpoint whose config also describes a classification head: the criterion now follows the task the estimator is built for rather than being inferred from `max_num_classes`. Loading a regression checkpoint into `TabPFNClassifier` now raises instead of silently building an unused bar distribution. ([#1204](https://github.com/PriorLabs/TabPFN/pull/1204))


## [8.4.0] - 2026-08-19

### Added

- `TabPFNRegressor` now accepts `eval_metric` and `tuning_config` arguments: passing `tuning_config={"calibrate_temperature": True}` makes `fit()` calibrate the temperature of the aggregated ensemble distribution on a holdout, sharpening or widening the predicted distribution as the data demands and improving every `predict()` output type, including the predicted quantiles. Set `eval_metric` to `"nll"` (the default, negative log-likelihood), `"crps"` (continuous ranked probability score, the same implementation used by the finetuning loss) to choose which quantity the calibration optimises; they weight the predicted distribution differently and pick noticeably different temperatures, so pick the metric you will be judged by. ([#1172](https://github.com/PriorLabs/TabPFN/pull/1172))
- In the previous setup, all GPUs in finetuning with DDP held all activations of all estimators in memory. This PR divides estimator activations across the available GPUs. ([#1182](https://github.com/PriorLabs/TabPFN/pull/1182))

### Changed

- The temperature grid searched when calibrating `TabPFNClassifier`'s softmax temperature now contains 1.0 exactly, so calibration can leave a distribution untouched. Previously the grid straddled 1.0 without including it, meaning a calibrated model always applied some correction even when none was warranted. Calibrated temperatures may therefore differ slightly from previous releases. ([#1172](https://github.com/PriorLabs/TabPFN/pull/1172))
- `fit()` cleans large tables with far less memory and time: the redundant float64 copies are gone, cutting both transient memory and wall time by about two thirds on a 5.3 GB all-numeric table. ([#1173](https://github.com/PriorLabs/TabPFN/pull/1173))
- `fit()` uses less memory on tables with categorical columns: the encoded array is now assembled in place instead of being stacked and then reordered, cutting transient memory by about a quarter on a half-string table. ([#1174](https://github.com/PriorLabs/TabPFN/pull/1174))
- `fit()` no longer slows to a crawl on wide tables with categorical columns under pandas < 3: the dtype casts no longer rebuild the frame one column at a time, which took minutes on a 333,333 x 400 table and now takes seconds. ([#1180](https://github.com/PriorLabs/TabPFN/pull/1180))
- Reduce peak GPU memory during KV-cache construction by quantizing layers as they are built and temporarily staging completed estimator caches on CPU in memory-saving mode. ([#1183](https://github.com/PriorLabs/TabPFN/pull/1183))

### Fixed

- `TabPFNRegressor.predict_batched` now raises `NotImplementedError` when the estimator was constructed with a `tuning_config`, instead of silently returning uncalibrated predictions. The ensemble temperature is calibrated on each dataset's own holdout, so a fused batch has no single temperature to apply; score such datasets individually with `predict`. This matches the existing guard in `TabPFNClassifier.predict_proba_batched`. ([#1172](https://github.com/PriorLabs/TabPFN/pull/1172))


## [8.3.0] - 2026-08-12

### Added

- Fine-tuning now supports `validation_frequency` to run validation and early-stopping checks every N epochs. ([#811](https://github.com/PriorLabs/TabPFN/pull/811))
- Add `ManyClassDecoder.attention_weights`, the canonical per-training-row attention distribution of the multiclass decoder head, so interpretability tooling can read out which training rows drive a prediction without reimplementing the head's internal forward pass. The method exists only on multiclass models that use this decoder. ([#1142](https://github.com/PriorLabs/TabPFN/pull/1142))
- Add fp8 kv cache dtype. ([#1157](https://github.com/PriorLabs/TabPFN/pull/1157))
- `fit()` now warns when a column of `X` looks like free text. ([#1159](https://github.com/PriorLabs/TabPFN/pull/1159))
- Add an opt-in built-model cache to `load_model`, enabled via the `TABPFN_MODEL_CACHE_SIZE` environment variable (default off). When set, repeated loads of the same checkpoint reuse the constructed model instead of rebuilding the architecture and re-running `load_state_dict`. Only the non-mutating (`cache_trainset_representation=False`) build is cached. ([#1162](https://github.com/PriorLabs/TabPFN/pull/1162))
- Add `TabPFNRegressor.predict_batched`, the regression counterpart to `TabPFNClassifier.predict_proba_batched`. It preprocesses each `(X_train, y_train, X_test)` triple exactly as `fit` + `predict` does, stacks the datasets on the model's batch dimension and scores them with a single fused forward per estimator, then decodes each dataset with its own target standardisation and per-estimator border transforms. Returns one entry per dataset in input order, each with the same structure `predict` would return for that dataset. Datasets must share array shapes; constant-target datasets are answered analytically.

  Both batched methods now raise `NotImplementedError` for `inference_precision=torch.float64` instead of silently computing the fused forward at float32 and returning float32-precision results. ([#1164](https://github.com/PriorLabs/TabPFN/pull/1164))
- Add an attention-backend registry; the in-tree FA3/torch-MPS/MLX paths now route through it. Behavior unchanged unless a backend is registered. ([#1165](https://github.com/PriorLabs/TabPFN/pull/1165))
- Add a test that `enable_torch_compile` traces tabpfn without graph breaks. ([#1166](https://github.com/PriorLabs/TabPFN/pull/1166))

### Changed

- Speed up cached prediction on Hopper GPUs with FlashAttention-3 installed, by splitting attention over the key/value sequence when few test rows attend over a large training cache. ([#1168](https://github.com/PriorLabs/TabPFN/pull/1168))
- `n_estimators` now defaults to `"auto"` on `TabPFNClassifier` and `TabPFNRegressor`. Feature-coverage auto-scaling (raising `n_estimators` on wide datasets so every feature is seen by some estimator) applies only to `"auto"` — an explicitly passed `n_estimators` is always used exactly as given, and warns at fit time if it is too small for every feature to be covered. ([#1171](https://github.com/PriorLabs/TabPFN/pull/1171))

### Fixed

- Fix activation checkpointing during v2, v2.5, and v2.6 fine-tuning after the state-container memory optimization. ([#1138](https://github.com/PriorLabs/TabPFN/pull/1138))
- - Fixed `fit()` crashing during temperature calibration or threshold tuning
    (`tuning_config`) when a rare class is absent from the tuning holdout, by
    passing the full label set to `log_loss` explicitly.
  - Fixed `fit()` crashing when `random_state` is a `np.random.Generator` and
    tuning is enabled, by converting the generator to a static seed before it
    reaches `StratifiedKFold`.

  ([#1140](https://github.com/PriorLabs/TabPFN/pull/1140))
- fixed the randomness in the truncated SVD to make runs more reproducible. ([#1167](https://github.com/PriorLabs/TabPFN/pull/1167))
- Fixed the many class decoder to only work with the relevant class count. ([#1175](https://github.com/PriorLabs/TabPFN/pull/1175))
- Skip the Claude code review workflow on bot-authored release PRs, which previously failed the `claude-review` check on every release. ([#1177](https://github.com/PriorLabs/TabPFN/pull/1177))

### Deprecated

- `auto_scale_n_estimators` is deprecated and will be removed in v9. It only ever applied to `n_estimators="auto"`, where `auto_scale_n_estimators=False` is equivalent to passing `n_estimators=8`; pass an explicit `n_estimators` instead to opt out of feature-coverage scaling. Passing `False` now emits a `FutureWarning` at fit time. ([#1171](https://github.com/PriorLabs/TabPFN/pull/1171))


## [8.2.0] - 2026-07-28

### Added

- Add `examples/input_gradients.py` showing how to extract gradients of TabPFN predictions with respect to the input data via `differentiable_input=True` on the breast-cancer dataset. ([#1128](https://github.com/PriorLabs/TabPFN/pull/1128))

### Changed

- Improved fine-tuning: retuned the example scripts (Higgs test AUC 0.8247 → 0.8322, California housing test MSE 0.1350 → 0.1328), `validation_split_ratio=None`/`0` now disables validation entirely, best-checkpoint saving now only applies with `early_stopping=True`, and small remainder data chunks no longer crash the context/query split. ([#1101](https://github.com/PriorLabs/TabPFN/pull/1101))
- Run the consistency tests with float64 inference precision and regenerate the reference predictions, reducing floating-point divergence across hardware/BLAS backends. ([#1111](https://github.com/PriorLabs/TabPFN/pull/1111))
- Expose the `kv_cache_dtype` as explicit argument. ([#1120](https://github.com/PriorLabs/TabPFN/pull/1120))
- `inference_precision="auto"` now uses bfloat16 autocast on CPUs with native bf16 support (Intel AMX / AVX512-BF16, AMD Zen 4+), giving ~2x faster CPU inference at unchanged accuracy. CPUs without fast bf16 keep running in float32. ([#1122](https://github.com/PriorLabs/TabPFN/pull/1122))
- Raised the default CPU sample limit from 1000 to 5000 for the v3 model (other versions keep the 1000 limit). With bf16 autocast on modern CPUs, v3 inference on datasets up to 5000 rows now runs in a reasonable time (~30s at 5000 rows). The "may be slow" warning threshold scales with the limit (1000 for v3, 200 otherwise). The override (`ignore_pretraining_limits=True` / `TABPFN_ALLOW_CPU_LARGE_DATASET=1`) is unchanged. ([#1123](https://github.com/PriorLabs/TabPFN/pull/1123))
- Enabled autocast also for mps. ([#1124](https://github.com/PriorLabs/TabPFN/pull/1124))
- Changed the minimal required version of torch to use MPS to 2.6 ([#1129](https://github.com/PriorLabs/TabPFN/pull/1129))
- Speed up preprocessing for large number of features ([#1136](https://github.com/PriorLabs/TabPFN/pull/1136))
- Speedup preprocessing for large numeric columns. ([#1137](https://github.com/PriorLabs/TabPFN/pull/1137))

### Fixed

- Fixed a `RuntimeWarning: overflow encountered in cast` in `SafePowerTransformer`'s
    Yeo-Johnson inverse transform, caused by the clip bound being computed in the
    output dtype instead of float64. ([#1105](https://github.com/PriorLabs/TabPFN/pull/1105))
- Fix timing on GPU to reflect full GPU work. ([#1113](https://github.com/PriorLabs/TabPFN/pull/1113))
- Fix `_repair_borders` widening the top bar-distribution border downwards for negative targets, which left the borders non-ascending when a quantile target transform collapsed the top gap. ([#1127](https://github.com/PriorLabs/TabPFN/pull/1127))


## [8.1.0] - 2026-07-13

### Added

- KV Cache support for 2.5 and 2.6 single file models. ([#1039](https://github.com/PriorLabs/TabPFN/pull/1039))
- Add `TabPFNClassifier.predict_proba_batched(X_list, y_list, X_test_list)` to score several independent datasets in a single fused forward per estimator (stacking them on the model's batch dimension), equivalent to fitting and predicting each dataset separately but much faster when launching many small predicts. ([#1045](https://github.com/PriorLabs/TabPFN/pull/1045))
- Add an opt-in `PASSTHROUGH_INF` inference-config option (default `False`), set via the `inference_config` argument of `TabPFNClassifier`/`TabPFNRegressor` (or, for the finetuned estimators, via the `inference_config` entry of their `extra_*_kwargs`). When enabled, `±inf` values are no longer rejected during `fit()`/`predict()`; they are carried through preprocessing (replaced with `NaN` for the steps that cannot handle them and restored afterwards) so they reach the model, which handles them natively. ([#1055](https://github.com/PriorLabs/TabPFN/pull/1055))
- README architecture and attention diagrams for TabPFN-3 (Prior Labs colour scheme). ([#1060](https://github.com/PriorLabs/TabPFN/pull/1060))
- Add `calculate_cache_size` for TabPFN v3 to compute the resident cache memory (ICL KV cache, decoder activations, distribution-embedder inducing states, and scaler stats) for a given train-set size, column count, ensemble size, and dtype — without running inference. ([#1087](https://github.com/PriorLabs/TabPFN/pull/1087))
- Add a public `tabpfn.finetuning.main_process_first()` context manager for multi-GPU (`torchrun`) scripts: the main process runs the with-block first while the other ranks wait at a barrier, then the other ranks run it — useful for one-time work such as dataset downloads that should warm a shared cache. The process group it initializes is reused by the subsequent `fit()`. ([#1094](https://github.com/PriorLabs/TabPFN/pull/1094))
- Chunk large test sets during cached (`fit_mode="fit_with_cache"`) inference to bound peak GPU memory, controlled by the new `TABPFN_MAX_BATCHED_TEST_ROWS` setting (default `32768`; set to `0` to disable). Chunking is mathematically equivalent. ([#1096](https://github.com/PriorLabs/TabPFN/pull/1096))

### Changed

- TabPFN-2 and TabPFN2.5 now use the single file implementation, deprecate 'base'. ([#1052](https://github.com/PriorLabs/TabPFN/pull/1052))
- Fine-tuning now targets the package default model version (`settings.tabpfn.model_version`) instead of a hardcoded older one, and `FinetunedTabPFNClassifier`/`FinetunedTabPFNRegressor` accept an optional `model_version` to override it — so a fine-tuned model is no longer silently compared against a different-generation base. ([#1064](https://github.com/PriorLabs/TabPFN/pull/1064))
- Reduce memory usage for v2.x architectures. Enable flash attention on MPS for v2_6. ([#1070](https://github.com/PriorLabs/TabPFN/pull/1070))

### Fixed

- Add `TabPFNRegressor.fit_with_differentiable_input(X, y)` so gradients can flow from a downstream loss back through the regressor into upstream torch modules feeding `X` (and `y`, when it carries grads). Mirrors the existing classifier-side path — previously `TabPFNRegressor.fit` raised `ValueError("Differentiable input is not supported for regressors yet.")` and there was no differentiable counterpart. ([#923](https://github.com/PriorLabs/TabPFN/pull/923))
- Support save/load for estimators fitted with `fit_mode="fit_with_cache"`. Previously `save_fit_state` / `load_from_fit_state` raised `NotImplementedError` for KV-cache inference engines. ([#977](https://github.com/PriorLabs/TabPFN/pull/977))
- Fix `save_fitted_tabpfn_model`/`save_fit_state` moving the live estimator's bar distribution modules to CPU, which broke subsequent `predict` calls (e.g. `output_type="median"`/`"quantiles"`) on CUDA/MPS devices. ([#1030](https://github.com/PriorLabs/TabPFN/pull/1030))
- Fixed `AdaptiveQuantileTransformer` losing `output_distribution` and `random_state` when cloned by sklearn (e.g. inside `ColumnTransformer.fit`), which made the `quantile_norm*` presets silently produce uniform output. All transformers now run sklearn's standard estimator checks. ([#1031](https://github.com/PriorLabs/TabPFN/pull/1031))
- Fixed `fit()` hanging forever when stratified row subsampling allocates a class more slots than it has rows (e.g. an ultra-rare class with `SUBSAMPLE_SAMPLES` set); such classes are now minimally oversampled instead. ([#1034](https://github.com/PriorLabs/TabPFN/pull/1034))
- Fixed `norm_and_kdi` returning a feature schema that undercounts the output columns: the FeatureUnion emits two columns per input column, so the schema and `num_added_features` under-reported, letting the ensemble's feature-budget planning silently exceed `max_features_per_estimator`. ([#1035](https://github.com/PriorLabs/TabPFN/pull/1035))
- Fixed an inverted `enable_gqa` condition in the torch-MPS attention fast path that would crash every forward of models with asymmetric query/KV head counts (including the default TabPFN v3 checkpoint) on Apple Silicon once torch satisfies the MPS flash-attention version gate (>= 2.13). ([#1037](https://github.com/PriorLabs/TabPFN/pull/1037))
- Fix fitted model saving for paths whose parent directories contain `.tabpfn_fit`. ([#1048](https://github.com/PriorLabs/TabPFN/pull/1048))
- Fix the README's save/load example to call `save_tabpfn_model(reg, ...)` with the estimator instead of `reg.model_`, which would have raised at runtime. ([#1053](https://github.com/PriorLabs/TabPFN/pull/1053))
- Remove all-NaN columns as constant features so they no longer leak NaNs into downstream preprocessing. ([#1061](https://github.com/PriorLabs/TabPFN/pull/1061))
- Fine-tuning with early stopping no longer returns a model worse than the base when no epoch improves over the default; the original weights are now restored. ([#1064](https://github.com/PriorLabs/TabPFN/pull/1064))
- Fix the README save/load FAQ to render correctly on GitHub (replace Sphinx `:func:` roles with code spans) and document the `TABPFN_MPS_MEMORY_FRACTION` environment variable. ([#1065](https://github.com/PriorLabs/TabPFN/pull/1065))
- Fix incorrect model output on MacOS 26 on M1 when using the MPS device. ([#1077](https://github.com/PriorLabs/TabPFN/pull/1077))
- Fix `predict_proba_batched` raising `RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float` under `inference_precision=torch.float16` on GPU. The batched inference engine now casts the model to the forced dtype, not just the inputs. ([#1083](https://github.com/PriorLabs/TabPFN/pull/1083))
- Fix the fine-tuning examples crashing or redundantly downloading their dataset once per rank when launched with `torchrun --nproc-per-node=N` on a cold sklearn cache; the dataset fetch is now wrapped in `main_process_first()` so only the main process downloads. ([#1094](https://github.com/PriorLabs/TabPFN/pull/1094))
- Fix cross-device save/load tests failing on GPU by only requiring functional equivalence, not bit-identical predictions, across devices. ([#1097](https://github.com/PriorLabs/TabPFN/pull/1097))
- Fix Windows CI crash (illegal instruction) by skipping the bfloat16 autocast KV-cache test on Windows without CUDA. ([#1103](https://github.com/PriorLabs/TabPFN/pull/1103))

### Deprecated

- Remove base architecture - per-model file implementations are now the single source of truth. ([#1056](https://github.com/PriorLabs/TabPFN/pull/1056))
- Remove the now unused InferenceEngineCacheKV. All models now use InferenceEngineExplicitKVCache. ([#1057](https://github.com/PriorLabs/TabPFN/pull/1057))


## [8.0.8] - 2026-06-10

### Breaking Changes

- Dropped support for Python 3.9; the minimum required version is now Python 3.10. The `eval-type-backport` dependency (only needed on 3.9) has been removed. ([#1038](https://github.com/PriorLabs/TabPFN/pull/1038))

### Added

- Add a single file implementation of TabPFNv2. Not activated by default yet. ([#995](https://github.com/PriorLabs/TabPFN/pull/995))
- Add a `keep_cache_on_device` option to `TabPFNClassifier`/`TabPFNRegressor` (defaults to `True`). When `fit_mode="fit_with_cache"`, setting it to `False` offloads each per-estimator KV cache to CPU as it is built and moves it back to the device on demand, lowering resident device memory at the cost of per-call transfers. ([#1009](https://github.com/PriorLabs/TabPFN/pull/1009))
- Added official support for Python 3.14 (already exercised by the CI test matrix). ([#1038](https://github.com/PriorLabs/TabPFN/pull/1038))

### Changed

- Improve peak memory of single file model implementations. ([#1019](https://github.com/PriorLabs/TabPFN/pull/1019))
- Removed the `per_feature` option from `PreprocessorConfig.name`. ([#1036](https://github.com/PriorLabs/TabPFN/pull/1036))

### Fixed

- Fixed regressor ensemble members sharing a single mutable `target_transform` instance. With in-process preprocessing (`n_preprocessing_jobs=1`), each member's in-place fit clobbered the fitted state of the others, silently corrupting predictions whenever members were fitted on different targets (e.g. with row subsampling active). Each ensemble config now owns a deep copy of the transform. ([#1029](https://github.com/PriorLabs/TabPFN/pull/1029))
- Fixed two GPU-preprocessing divergences from the CPU reference: `TorchSoftClipOutliers` silently skipped outlier clipping when predicting a single sample in KV-cache mode (predictions depended on test batch size), and `TorchAddSVDFeaturesStep` added an SVD column for single-feature datasets where the CPU pipeline adds none (predictions differed between `ENABLE_GPU_PREPROCESSING` on and off). ([#1033](https://github.com/PriorLabs/TabPFN/pull/1033))
- Fixed a fit-time crash when a DataFrame mixed a plain numpy `bool` column with a non-numeric string column (string-valued `category` or pandas `string` dtype). `coerce_nullable_dtypes_to_numpy` now coerces numpy `bool` columns to float64, not only nullable extension dtypes. ([#1040](https://github.com/PriorLabs/TabPFN/pull/1040))


## [8.0.7] - 2026-06-08

### Changed

- At predict time, an encoded column whose dtype differs from fit is now coerced to its fit-time dtype (and warns). For a numeric-categorical column arriving as strings, numeric-looking strings (`"1.0"`) now match their fit category instead of all being treated as unseen. ([#1015](https://github.com/PriorLabs/TabPFN/pull/1015))

### Fixed

- Fix a crash in the chunked-inference OOM recovery path that called `torch.mps.empty_cache()` unconditionally, raising `Cannot execute emptyCache() without MPS backend` on non-MPS devices (CUDA GPUs, CPU-only Linux) and turning a recoverable out-of-memory into a hard failure. ([#1007](https://github.com/PriorLabs/TabPFN/pull/1007))
- Fixed two crashes from inconsistent column dtypes: `fit` raising `Cannot cast object dtype to float64` when a nullable extension dtype (`Int64`/`Float64`/`boolean`) sits next to a string categorical column, and `predict` raising a `TypeError` when a column was string/categorical at fit but arrives numeric. ([#1015](https://github.com/PriorLabs/TabPFN/pull/1015))


## [8.0.6] - 2026-06-03

### Added

- Add `auto_scale_n_estimators` constructor argument (default `True`) to auto-scale `n_estimators` for full feature coverage on wide datasets, capped at 32. ([#1000](https://github.com/PriorLabs/TabPFN/pull/1000))


## [8.0.5] - 2026-06-03

### Fixed

- Fixed a `could not convert string to float` crash when a feature declared categorical via `categorical_features_indices` is all-missing during fit but has real string values at predict. Such columns are now kept categorical instead of being demoted to a constant numeric column, so they route through the ordinal encoder consistently between fit and predict. ([#1002](https://github.com/PriorLabs/TabPFN/pull/1002))


## [8.0.4] - 2026-06-03

### Added

- Add SafeTensors checkpoint loading. TabPFN can now load model checkpoints from `.safetensors` files in addition to the legacy `.ckpt` format, with non-tensor metadata (architecture name, model config, inference config) embedded in the safetensors header. ([#981](https://github.com/PriorLabs/TabPFN/pull/981))
- Register `tabpfn-v3-classifier-v3_20260506_ood.ckpt` and `tabpfn-v3-regressor-v3_20260506_ood.ckpt` so they can be loaded from Hugging Face by filename. ([#982](https://github.com/PriorLabs/TabPFN/pull/982))
- Add a visualisation utility to plot the predicted distribution (regression) in `tabpfn.visualization` ([#987](https://github.com/PriorLabs/TabPFN/pull/987))

### Changed

- Remove the feature selection cell from the TabPFN_Demo_Local example notebook. ([#978](https://github.com/PriorLabs/TabPFN/pull/978))
- Quantize KV cache to int8 for `fit_mode="fit_with_cache"` on TabPFN-3 models. Reduces ICL KV cache memory ~2 with no accuracy loss. ([#983](https://github.com/PriorLabs/TabPFN/pull/983))

### Fixed

- Fixed a `could not convert string to float` crash when a categorical/string feature is all-missing during fit but has real string values at predict, caused by a fit/predict dtype-routing asymmetry in the ordinal encoder. ([#992](https://github.com/PriorLabs/TabPFN/pull/992))


## [8.0.3] - 2026-05-16

### Changed

- Significantly reduced `import tabpfn` time (roughly halved: ~2.4s → ~1.1s warm, and ~9s → ~5s on a cold first import) by no longer importing `torch._dynamo`/`torch._inductor` or scikit-learn's estimator-check test machinery at import time. ([#972](https://github.com/PriorLabs/TabPFN/pull/972))


## [8.0.2] - 2026-05-13

### Added

- Add flash attention support for MPS to reduce memory usage. Remove attention_backend. ([#949](https://github.com/PriorLabs/TabPFN/pull/949))

### Changed

- Modernized the SHAP / Shapley Values section in `TabPFN_Demo_Local.ipynb` to use `shapiq` (with TabPFN's KV cache enabled), and made small fixes to the feature-selection, time-series, and causal-inference sections. ([#960](https://github.com/PriorLabs/TabPFN/pull/960))


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
