# Telemetry

TabPFN includes lightweight, optional telemetry that helps us understand how the library is used and where to focus development. This page explains exactly what is collected, how it's handled, and how to opt out.

## What we collect

We gather high-level usage signals - enough to guide development, never enough to expose your data or code.

**Events**

- `session` - sent when a TabPFN estimator is initialized
- `ping` - liveness check on model initialization
- `model_load` - sent when a model is loaded from disk or cache
- `fit_called` / `predict_called` - sent when you call `fit` or `predict`

**Metadata (all events)**

- `tabpfn_version`, `python_version`, `numpy_version`, `pandas_version` - software versions
- `gpu_type` - GPU type TabPFN is running on
- `timestamp` - time of the event
- `install_date` - date TabPFN was first used (year-month-day)
- `install_id` - random, locally generated installation identifier (see "Privacy" below)

**Additional metadata (fit / predict only)**

- `task` - classification or regression
- `num_rows`, `num_columns` - dataset shape, rounded into ranges (exact values are never recorded)
- `duration_ms` - wall-clock time of the call

## What we never collect

Regardless of account status, we never collect:

- Training data, features, labels, or model outputs
- File paths, environment variables, or hostnames
- Exact dataset dimensions
- Code of any kind

No inputs, outputs, or model weights ever leave your machine.

## Privacy

TabPFN operates in two modes with different privacy properties:

**Without an account (anonymous).** Telemetry is tied only to a random `install_id` generated locally on first use. This ID is not linked to any personal information and cannot be traced back to you.

**With an account (pseudonymous).** If you create a TabPFN account, your `user_id` is included in telemetry events. 

For further details we suggest you check out our [privacy policy](https://priorlabs.ai/privacy-policy).

## Opting out

Set one environment variable to disable all telemetry:

```bash
export TABPFN_DISABLE_TELEMETRY=1
```

## Why collect telemetry?

Open-source projects get limited feedback unless people file issues. Telemetry helps us see which parts of TabPFN are most used, detect performance bottlenecks, and prioritize improvements that benefit the most users.
