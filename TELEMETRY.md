# 📊 Telemetry

This project includes lightweight, anonymous telemetry to help us improve TabPFN.  
We've designed this with two goals in mind:

1. ✅ Be **fully GDPR-compliant** (no personal data, no sensitive data, no surprises)  
2. ✅ Be **OSS-friendly and transparent** about what we track and why  

If you'd rather not send telemetry, you can always opt out (see **Opting out**).

---

## 🔍 What we collect

We only gather **very high-level usage signals** — enough to guide development, never enough to identify you or your data.  

Here's the full list:

### Events
- `ping` – sent when models initialize, used to check liveness  
- `fit_called` – sent when you call `fit`  
- `predict_called` – sent when you call `predict`  

### Metadata (all events)
- `python_version` – version of Python you're running  
- `tabpfn_version` – TabPFN package version  
- `timestamp` – time of the event  

### Extra metadata (`fit` / `predict` only)
- `task` – whether the call was for **classification** or **regression**  
- `num_rows` – *rounded* number of rows in your dataset  
- `num_columns` – *rounded* number of columns in your dataset  
- `duration_ms` – time it took to complete the call  

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