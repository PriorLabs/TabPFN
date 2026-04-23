# 📊 Telemetry

This project includes lightweight, anonymous telemetry to help us improve TabPFN. If you'd rather not send telemetry, you can always opt out (see **Opting out**).

---

## 🔍 What we collect

We only gather **very high-level usage signals** — enough to guide development, never enough to identify you or your data.

Here's the full list:

### Events
- `ping` – periodic liveness heartbeat (daily / weekly / monthly cadence)
- `session` – sent when you initialize a TabPFN estimator (`TabPFNClassifier`, `TabPFNRegressor`)
- `model_load` – sent when TabPFN attempts to load model weights (reports `success` / `failed`)
- `dataset` – sent when a dataset is passed to `fit` or `predict` (no dataset content; shape only)
- `fit_called` – sent when you call `fit`
- `predict_called` – sent when you call `predict`
- `extension_entry` – sent when a TabPFN extension entry point (e.g. from `tabpfn-extensions`, `tabpfn-time-series`) is invoked

### Metadata (all events)
- `python_version` – Python version you're running
- `tabpfn_version` – TabPFN package version
- `numpy_version` – local NumPy version
- `pandas_version` – local pandas version
- `gpu_type` – type of GPU TabPFN is running on
- `platform_os` – operating system
- `runtime_kernel` – runtime kernel (e.g. CPython)
- `runtime_environment` – runtime environment (e.g. notebook / script / CI)
- `timestamp` – time of the event
- `install_date` – `year-month-day` when TabPFN was used for the first time
- `install_id` – unique, random and anonymous installation ID

### Extra metadata (per-event)
- `fit_called` / `predict_called`: `task` (classification or regression), `num_rows` (*rounded*), `num_columns` (*rounded*), `duration_ms`
- `model_load`: `model_name` (HuggingFace repo id), `status`
- `dataset`: `task`, `role` (train / test), `num_rows` (*rounded*), `num_columns` (*rounded*)
- `extension_entry`: `extension_name`

---

## 🛡️ How we protect your privacy

- **No inputs, no outputs, no code** ever leave your machine.
- **No personal data** is collected.
- Dataset shapes are **rounded into ranges** (e.g. `(953, 17)` → `(1000, 20)`) so exact dimensionalities can't be linked back to you.
- The data is strictly anonymous — it cannot be tied to individuals, projects, or datasets.

This approach lets us understand dataset *patterns* (e.g. "most users run with ~1k features") while ensuring no one's data is exposed.

---

## 🤔 Why collect telemetry?

Open-source projects don't get much feedback unless people file issues. Telemetry helps us:
- See which parts of TabPFN are most used (fit vs predict, classification vs regression)
- Detect performance bottlenecks and stability issues
- Prioritize improvements that benefit the most users

This information goes directly into **making TabPFN better** for the community.

---

## 🚫 Opting out

Don't want to send telemetry? No problem — just set the environment variable:

```bash
export TABPFN_DISABLE_TELEMETRY=1
```
