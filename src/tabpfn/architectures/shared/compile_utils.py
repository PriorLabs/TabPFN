#  Copyright (c) Prior Labs GmbH 2026.

"""Lazy helpers for explicitly selected compilation regions.

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
import threading
from collections.abc import Callable
from typing import Any, TypeVar

import torch
from torch.torch_version import TorchVersion

F = TypeVar("F", bound=Callable[..., Any])


def compile_when_enabled(fn: F) -> F:
    """Compile a tensor region when its ``enable_torch_compile`` keyword is true.

    The decorated function must accept that keyword, defaulting to False. Its
    callers and all undecorated functions stay eager. Compiler initialization is
    lazy, and the cached callable takes the module as an ordinary argument, so
    no compiled callables or compilation settings enter model state/pickles.

    Tensor arguments must have leading batch/sequence dimensions and a final
    embedding dimension. Mark the leading dimensions independently dynamic to
    avoid accidental equality guards when, for example, batch size initially
    equals the number of CLS tokens. Model embedding/parameter sizes stay static.
    """
    compiled: Callable[..., Any] | None = None
    init_lock = threading.Lock()

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal compiled
        if not kwargs.get("enable_torch_compile", False):
            return fn(*args, **kwargs)
        if TorchVersion(torch.__version__) < TorchVersion("2.6"):
            raise ValueError(
                "v3.5 compilation requires PyTorch >= 2.6. Upgrade PyTorch or "
                "set enable_torch_compile=False."
            )
        for arg in (*args, *kwargs.values()):
            if isinstance(arg, torch.Tensor):
                torch._dynamo.mark_dynamic(arg, list(range(arg.ndim - 1)))
        if compiled is None:
            with init_lock:
                if compiled is None:
                    compiled = torch.compile(fn, dynamic=True, fullgraph=False)
        return compiled(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


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
