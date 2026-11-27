# Third-Party Notices

This file documents third-party code adapted into this repository, with
upstream attribution preserved. Transitive dependencies installed via
`pip` are governed by their own licenses (see `pyproject.toml` for the
canonical list).

---

## Summary

| Upstream | Local path | Upstream license |
|---|---|---|
| skrub — `SquashingScaler` | `src/tabpfn/preprocessing/steps/squashing_scaler_transformer.py`<br>`src/tabpfn/preprocessing/torch/torch_squashing_scaler.py` | BSD-3-Clause |

---

## Per-upstream notices

### skrub — SquashingScaler

**Upstream:** https://github.com/skrub-data/skrub
**Local paths:**
- `src/tabpfn/preprocessing/steps/squashing_scaler_transformer.py` — CPU/scikit-learn implementation
- `src/tabpfn/preprocessing/torch/torch_squashing_scaler.py` — PyTorch port of the same algorithm, with explicit-state fit/apply semantics

**License:** BSD-3-Clause
**Copyright:** Copyright (c) 2018-2023, The dirty_cat developers, 2023-2026 the skrub developers. All rights reserved. (per the skrub `LICENSE.txt`)
**Modifications:** Adapted to fit TabPFN's preprocessing pipeline; algorithmic logic preserved across both implementations. Upstream does not ship a per-file copyright header; attribution is carried in this NOTICE plus the in-file blocks.

---

## Adding new entries

When vendoring or adapting third-party code:

1. Preserve any upstream per-file copyright and license header verbatim. If the upstream does not ship a per-file header, add an attribution block citing the upstream URL, copyright holder, and SPDX license identifier.
2. When vendoring a whole directory of upstream code, also vendor the upstream `LICENSE` / `NOTICE` file alongside it. For single-file adaptations, the in-file attribution plus the entry in this NOTICE file is sufficient.
3. Add a row to the summary table and a per-upstream notice to this file, including the upstream copyright line when one is published.
