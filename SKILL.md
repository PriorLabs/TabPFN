---
name: tabpfn
description: Use TabPFN (TabPFNClassifier / TabPFNRegressor) to make predictions on tabular data. Covers licence and login, model choice, raw-data input with categorical columns, large datasets, fast prediction, memory errors and interpretability. Read this before writing any code that imports tabpfn.
---

# Using TabPFN

TabPFN is a pretrained model for tabular classification and regression. `fit`
does not train weights: it stores the training rows as context, and `predict`
runs one forward pass over context and test rows. It has a scikit-learn
interface.

This file is for agents that *use* TabPFN. If you are changing this repository,
read [`AGENTS.md`](AGENTS.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md) instead.

Source of truth: this file states rules and points to code. Numbers (limits,
defaults, versions) live in the checkpoints, the docstrings of
[`classifier.py`](src/tabpfn/classifier.py) and
[`regressor.py`](src/tabpfn/regressor.py), and the [README](README.md). Read
them there, do not copy them from memory. The Prior Labs documentation is also
good to check, through the docs MCP server (`https://docs.priorlabs.ai/mcp`).

## 1. Licence and login: do not work around them

- The code is Apache 2.0. The **model weights** have their own licences. The
  default weights (TabPFN-2.5 and later) are **non-commercial**. TabPFN-2
  weights use the Prior Labs License (Apache 2.0 + attribution). The licence
  links are in the README "License" section.
- **What the non-commercial licence allows** (details: the "license" FAQ on the
  [Models page](https://docs.priorlabs.ai/models)): free internal use for
  research and exploration. That includes testing on the user's own data and
  **benchmarking TabPFN against the user's own corporate baselines** (for
  example XGBoost or an in-house model). The same holds for fine-tuning and for
  the model's outputs: non-commercial evaluation only.
- **What needs a commercial licence or the API agreement:** use in production
  or in a product, in workflows that support business decisions, in client
  projects or revenue-generating work, competitive benchmarking for procurement
  or vendor selection, and using TabPFN or its outputs to train, distill or
  improve another model for commercial use. If the user's task falls here, tell
  them and point them to https://priorlabs.ai/pricing or sales@priorlabs.ai.
  Do not decide this for them and do not hide it.
- The first `fit` downloads the weights and checks that the user accepted the
  licence (`src/tabpfn/browser_auth.py`). With a browser, a login window opens.
  Without one (CI, SSH, notebooks on a server), `fit` raises
  `TabPFNLicenseError` with the steps to follow.
- **Accepting the licence is the human's decision.** Stop and ask the user to:
  1. log in at https://ux.priorlabs.ai and accept the licence on the Licenses tab,
  2. copy their API key from https://ux.priorlabs.ai/account,
  3. set `TABPFN_TOKEN` in the environment (or a `.env` file).
- The token is then cached in the user cache directory. A token from
  `tabpfn-client` is reused.
- **Never**: patch or monkey-patch the licence check, download weights from a
  mirror to skip it, write a fake token, or silently switch to the TabPFN-2
  weights to avoid the login. If the user cannot log in, tell them and stop.
- Offline use after the first download, `TABPFN_MODEL_CACHE_DIR` and
  `model_path`: see the README FAQ "How do I use TabPFN without an internet
  connection?".

## 2. Install and hardware

- `pip install tabpfn` (latest release). Python 3.10+.
- Use a GPU. On CPU, `fit` refuses datasets above the checkpoint's CPU limit
  (`MAX_CPU_SAMPLES`). Do not set `TABPFN_ALLOW_CPU_LARGE_DATASET` or
  `ignore_pretraining_limits=True` to get around it; tell the user it needs a
  GPU, or use the hosted API with
  [`tabpfn-client`](https://github.com/PriorLabs/tabpfn-client) (this sends
  the data to Prior Labs servers, so ask the user first). The servers are
  safe; the [security page](https://docs.priorlabs.ai/api/security.md) has the
  details (encryption, isolation, access control).

## 3. Model choice

- `TabPFNClassifier()` / `TabPFNRegressor()` load the current default model.
  Keep the default unless the user asks for another one.
- Other versions: `TabPFNClassifier.create_default_for_version(ModelVersion.X)`
  with `from tabpfn.constants import ModelVersion`. The list of versions is the
  `ModelVersion` enum in [`constants.py`](src/tabpfn/constants.py); the
  checkpoint files per version are in `ModelSource` in
  [`model_loading.py`](src/tabpfn/model_loading.py).
- Limits (rows, features, classes) and default settings are stored **in the
  checkpoint**. Read them from a fitted model instead of guessing:

  ```python
  cfg = model.inference_config_  # after fit
  print(cfg.MAX_NUMBER_OF_SAMPLES, cfg.MAX_NUMBER_OF_FEATURES,
        cfg.MAX_NUMBER_OF_CLASSES, cfg.MAX_CPU_SAMPLES, cfg.N_ESTIMATORS)
  ```

  Never invent a row limit and never cut the data to fit an imagined one.

## 4. Input data: pass raw data, flag categoricals

- Pass a pandas `DataFrame` with the raw columns. TabPFN handles missing
  values, strings, dates and text itself. Do **not** scale, impute, one-hot
  encode or remove outliers.
- **Never turn categorical columns into integer codes passed as numbers**
  (`LabelEncoder`, `OrdinalEncoder`, `.cat.codes`, `factorize`, ...). The model
  then reads them as ordered quantities. Keep the raw values and **declare the
  categorical columns** with `categorical_features_indices` (0-based positions
  in the `X` given to `fit`). A pandas `category` dtype column counts as
  declared too.
- Columns that hold numbers but mean categories (store id, zip code, product
  code) must also be declared, with their raw values.

```python
import pandas as pd
from tabpfn import TabPFNClassifier

X_train = pd.DataFrame({
    "age": [34, 51, 29, 46],
    "income": [42_000.0, None, 31_500.0, 58_000.0],  # missing value: keep it
    "city": ["Berlin", "Paris", "Berlin", "Rome"],     # string category
    "store_id": [101, 205, 101, 330],                  # numeric code, still a category
})
y_train = [0, 1, 0, 1]

cat_cols = ["city", "store_id"]
clf = TabPFNClassifier(
    categorical_features_indices=[X_train.columns.get_loc(c) for c in cat_cols],
)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)  # X_test: same columns, same order
```

- Same for `TabPFNRegressor`.
- Use the same column order in `fit` and `predict`. If you change the columns
  after creating the estimator, set `model.categorical_features_indices` again
  before `fit`.
- Grouped rows (several rows per patient, customer, ...): see `group_col` in
  the constructor docstring.

## 5. Large datasets: native subsampling only

- First fit on **all** training rows if they are within the checkpoint limit
  (section 3).
- If the data is above the limit, or fit/predict is too slow or runs out of
  memory, use TabPFN's own per-estimator subsampling. Each estimator then sees
  a different subsample, so the ensemble still covers the data:

  ```python
  TabPFNRegressor(inference_config={"SUBSAMPLE_SAMPLES": n_rows_per_estimator})
  ```

  `SAMPLE_SUBSAMPLING_METHOD` chooses how rows are drawn (`"balanced"`,
  `"stratified"`, `"majority_downsample"`); see its docstring in
  [`inference_config.py`](src/tabpfn/inference_config.py).
- **Never write your own sampler**: no `df.sample(...)` before `fit`, no loop
  that fits one model per random chunk and averages, no manual partitions.

## 6. Prediction speed

- Fit once. Pass the **whole test set in one `predict` call**. Predicting row
  by row is many times slower.
- If you predict several times against the same training data, use
  `fit_mode="fit_with_cache"`. TabPFN then splits large test sets into chunks
  itself. Do not write manual chunk loops over the test set.
- Regression outputs: `predict(X, output_type=...)` gives `"mean"`,
  `"median"`, `"mode"`, `"quantiles"` (with `quantiles=[...]`), `"main"` or
  `"full"`. Choose the one that matches the user's metric (for example
  `"median"` for MAE).
- Nonnegative targets with many exact zeros (claims, demand): consider
  `TabPFNRegressor(presets="zero_inflated")`; `fit` warns when it would help.

## 7. Out of memory

1. Retry the same fit with `memory_saving_mode=True`.
2. Then use a GPU with more memory, or native `SUBSAMPLE_SAMPLES` (section 5).
3. Then ask the user. Do not lower internal batch sizes, change
   `PYTORCH_CUDA_ALLOC_CONF`, or write custom chunking.

## 8. Saving, interpretability, more

- Save and reload a fitted model: `save_fitted_tabpfn_model` /
  `load_fitted_tabpfn_model` from `tabpfn.model_loading` (README FAQ).
- Interpretability (SHAP, Shapley interactions, partial dependence, feature
  selection): `pip install tabpfn-extensions`, module
  `tabpfn_extensions.interpretability`
  ([repo](https://github.com/PriorLabs/tabpfn-extensions)).
- Input gradients: [`examples/input_gradients.py`](examples/input_gradients.py).
- More than the checkpoint's class limit: `tabpfn_extensions.many_class`.
- Fine-tuning, prompt tuning, threshold tuning: see [`examples/`](examples).
- Environment variables: README FAQ "What environment variables can I use to
  configure TabPFN?".
- More examples: the Prior Labs docs MCP server (`https://docs.priorlabs.ai/mcp`).

## Checklist before you run TabPFN code

- [ ] The user accepted the licence themselves; no licence workaround in the code.
- [ ] Raw DataFrame in, no scaling / imputation / one-hot.
- [ ] No categorical column integer-encoded and passed as numeric; all
      categorical columns are in `categorical_features_indices`.
- [ ] No home-made subsampling (`sample(`, manual chunks, per-chunk models);
      only `SUBSAMPLE_SAMPLES` if needed.
- [ ] One `fit`, one `predict` per test set; `fit_mode="fit_with_cache"` for
      repeated predictions.
- [ ] Running on a GPU; on OOM, `memory_saving_mode=True` first.
