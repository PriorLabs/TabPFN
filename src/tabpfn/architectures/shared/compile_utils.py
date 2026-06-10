#  Copyright (c) Prior Labs GmbH 2026.

"""Lazy wrapper around ``torch.compiler.disable``.

Applying ``@torch.compiler.disable`` at class-definition time forces
``torch._dynamo`` / ``torch._inductor`` to be imported during ``import
tabpfn`` (~0.5s, hundreds of submodules) -- even though ``torch.compile`` is
opt-in (``PerformanceOptions.enable_torch_compile``, default ``False``) and the
disabled methods only need special handling while compiling.

``lazy_compiler_disable`` defers that machinery until compilation actually
runs. Merely *referencing* ``torch.compiler.disable`` does not import dynamo;
only *applying* it does, so ``import torch`` at module scope here is safe.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import torch

F = TypeVar("F", bound=Callable[..., Any])


def lazy_compiler_disable(fn: F) -> F:
    """``torch.compiler.disable`` as a decorator, applied lazily.

    ``torch.compiler.disable`` only has an effect while running under
    ``torch.compile``; in eager mode it is equivalent to calling ``fn``
    directly. We use that to avoid importing dynamo entirely in the common
    (eager) case:

    * **Not compiling** -> call ``fn`` directly. Behaviourally identical to
      ``torch.compiler.disable`` and never imports ``torch._dynamo``. This is
      the path taken by every normal ``predict``/``fit``.
    * **Compiling** (``torch.compiler.is_compiling()`` is true) -> build &
      cache the real ``torch.compiler.disable``-wrapped callable on first use.
      Dynamo graph-breaks when it traces the ``torch.compiler.disable`` call,
      so ``fn`` runs eagerly (exactly what the decorator guarantees) even when
      the very first call happens under compile.

    Verified by tests for the eager, eager-then-compile, and
    first-call-under-compile cases.
    """
    disabled: Callable[..., Any] | None = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal disabled
        if not torch.compiler.is_compiling():
            return fn(*args, **kwargs)
        if disabled is None:
            disabled = torch.compiler.disable(fn)
        return disabled(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
