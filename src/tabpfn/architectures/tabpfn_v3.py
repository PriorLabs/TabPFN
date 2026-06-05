# ruff: noqa: PLR0912, C901,
"""TabPFN v3 architecture.


Shape suffix convention:

B: batch size
R: total rows (train + test)
Ri: input rows, could be either train + test or test.
Rj: Chunked rows (<= R)
N: train rows
M: test rows
C: total columns
Cj: Chunked columns (<= C)
E: embedding dimension
T: Target dim (e.g. number of classes).
Cl: number of CLS tokens

D: head dimension
H: num heads
S: sequence length

Copyright (c) Prior Labs GmbH 2026.
"""

from __future__ import annotations

import dataclasses
import logging as _logging
import math
