# 📊 Telemetry

This project includes lightweight telemetry to help us improve TabPFN.  
We've designed this with two goals in mind:

1. ✅ Be **fully GDPR-compliant** (transparent, opt-out at any time, no surprises)
2. ✅ Be **OSS-friendly** about what we track and why

---

## 🔍 What we collect

We gather **high-level usage signals** - enough to guide development, never enough to expose your data or code.

### Events
- `ping` – sent when models initialize, used to check liveness
- `fit_called` – sent when you call `fit`
- `predict_called` – sent when you call `predict`
- `session` – sent whenever a TabPFN estimator is initialized

### Metadata (all events)
- `python_version` – version of Python you're running
- `tabpfn_version` – TabPFN package version
- `timestamp` – time of the event
- `numpy_version` – local NumPy version
- `pandas_version` – local Pandas version
- `gpu_type` – type of GPU TabPFN is running on
- `install_date` – `year-month-day` when TabPFN was used for the first time
- `install_id` – unique, random installation identifier (see below)

### Extra metadata (`fit` / `predict` only)
- `task` – classification or regression
- `num_rows` – *rounded* number of rows in your dataset
- `num_columns` – *rounded* number of columns in your dataset
- `duration_ms` – time it took to complete the call

---

## 👤 Anonymous vs. account-linked telemetry

TabPFN operates in two modes with **different privacy properties**:

### Without an account (anonymous)

If you use TabPFN without logging in, telemetry is tied only to a random 
`install_id` generated locally on first use. This ID is **not linked to any 
personal information** and cannot be traced back to you.

### With an account (pseudonymous)

If you accept the TabPFN license and create an account, your `user_id` is 
included in telemetry events. This means:

- Usage data (events and metadata listed above) **is linked to your account**
- This constitutes **processing of personal data** under GDPR
- Prior Labs GmbH acts as the **data controller** for this data
- Our legal basis is **legitimate interests** (Article 6(1)(f) GDPR): understanding how our model is used in order to improve it. We do not use this data for advertising or share it with third parties.

Your **data subject rights** apply in full: you may request access, correction, 
deletion, or export of your telemetry data at any time by contacting hello@priorlabs.ai.

Even with an account, **no inputs, outputs, model weights, or code ever leave 
your machine.** Dataset shapes are rounded into ranges so exact dimensionalities 
are not recorded.

---

## 🛡️ What we never collect

Regardless of account status:

- No training data, features, labels, or model outputs
- No file paths, environment variables, or hostnames
- No exact dataset shapes (only rounded ranges)
- No code

---

## 🚫 Opting out

Set the environment variable to disable all telemetry entirely:
```bash
export TABPFN_DISABLE_TELEMETRY=1
```

---

## 🤔 Why collect telemetry?

Open-source projects don't get much feedback unless people file issues. 
Telemetry helps us:

- See which parts of TabPFN are most used
- Detect performance bottlenecks
- Prioritize improvements that benefit the most users

This goes directly into **making TabPFN better** for the community.