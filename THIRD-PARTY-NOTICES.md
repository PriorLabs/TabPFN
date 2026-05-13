# Third-Party Notices

This file documents third-party code adapted into this repository, with
upstream attribution preserved. Transitive dependencies installed via
`pip` are governed by their own licenses (see `pyproject.toml` for the
canonical list).

---

## Summary

| Upstream | Local path | Upstream license |
|---|---|---|
| skrub — `SquashingScaler` | `src/tabpfn/preprocessing/steps/squashing_scaler_transformer.py` | BSD-3-Clause |

---

## Per-upstream notices

### skrub — SquashingScaler

**Upstream:** https://github.com/skrub-data/skrub
**Local path:** `src/tabpfn/preprocessing/steps/squashing_scaler_transformer.py`
**License:** BSD-3-Clause
**Modifications:** Adapted to fit TabPFN's preprocessing pipeline; algorithmic logic preserved.

---

## Adding new entries

When vendoring or adapting third-party code:

1. Preserve the upstream copyright and license header verbatim at the top of the affected source file (for whole-file vendoring) or inline next to the adapted block (for partial adaptation).
2. If the upstream ships a `LICENSE` / `NOTICE` file, vendor that file alongside the code.
3. Add a row to the summary table and a per-upstream notice to this file.
