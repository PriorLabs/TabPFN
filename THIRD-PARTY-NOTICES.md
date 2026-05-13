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
| sklearn-compat — single-file compat shim | `src/tabpfn/misc/_sklearn_compat.py` | BSD-3-Clause |
| SciPy — `_yeojohnson` from scipy PR #18852 | `src/tabpfn/preprocessing/steps/safe_power_transformer.py` (function `_yeojohnson`) | BSD-3-Clause |

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

### sklearn-compat — compatibility shim

**Upstream:** https://github.com/sklearn-compat/sklearn-compat
**Local path:** `src/tabpfn/misc/_sklearn_compat.py` (single-file vendored distribution; ~1000 lines)
**License:** BSD-3-Clause
**Copyright:** Copyright (c) 2024, Guillaume Lemaitre (per the upstream `LICENSE`)
**Modifications:** None of significance; vendored verbatim from upstream version 0.1.4 to avoid an extra runtime dependency. The single-file format is the distribution model sklearn-compat itself encourages for downstream users.

### SciPy — `_yeojohnson` overflow fix

**Upstream:** https://github.com/scipy/scipy/pull/18852
**Local path:** `src/tabpfn/preprocessing/steps/safe_power_transformer.py` (function `_yeojohnson`)
**License:** BSD-3-Clause
**Copyright:** Copyright (c) 2001-2002 Enthought, Inc.; 2003-2025 SciPy Developers.
**Modifications:** Adapted from the PR; one line changed (`return (x, None) if lmbda is None else x` instead of `return x` when input is empty) to align with the local call site. Once support for scipy < 1.12 is dropped, this can be replaced with a direct `scipy.stats.yeojohnson` import.

---

## Adding new entries

When vendoring or adapting third-party code:

1. Preserve any upstream per-file copyright and license header verbatim. If the upstream does not ship a per-file header, add an attribution block citing the upstream URL, copyright holder, and SPDX license identifier.
2. When vendoring a whole directory of upstream code, also vendor the upstream `LICENSE` / `NOTICE` file alongside it. For single-file adaptations, the in-file attribution plus the entry in this NOTICE file is sufficient.
3. Add a row to the summary table and a per-upstream notice to this file, including the upstream copyright line when one is published.
