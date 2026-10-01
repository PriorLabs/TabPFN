#  Copyright (c) Prior Labs GmbH 2026.

"""The `log_usage` decorator, and where the usage events it logs go.

Usage is logged only for users who have opted in to usage telemetry, for whom a
sink is installed with `set_sink`. For everyone else, which is the default, no
usage event is built, collected or sent.
"""

from __future__ import annotations

import functools
import inspect
import logging
import os
import time
from collections.abc import Callable, Mapping
from contextvars import ContextVar, copy_context
from datetime import datetime, timezone
from typing import Any, Concatenate, ParamSpec, TypeVar

from tabpfn.telemetry.events import EventKind, usage_event
from tabpfn.telemetry.parameters import label_of

logger = logging.getLogger(__name__)

_EstimatorT = TypeVar("_EstimatorT")
_P = ParamSpec("_P")
_R = TypeVar("_R")

# Receives every usage event; None while nothing is logged. All threads share
# it, and reading or replacing it is atomic, so it needs no lock.
_sink: Callable[[dict[str, Any]], None] | None = None

# Set while the current thread, or asyncio task, is inside a logged call. Each
# thread and task has its own value, so calls on other threads never see it. It
# says nothing about any estimator: everything logged is read from the call's
# own estimator and arguments, so estimators cannot overwrite each other's data.
_in_logged_call: ContextVar[bool] = ContextVar("tabpfn_in_logged_call", default=False)


def set_sink(sink: Callable[[dict[str, Any]], None] | None) -> None:
    """Install the function that receives every usage event.

    The sink is called on the thread that made the call, right after the call
    finishes, so it must be thread-safe and must not block. A call finishing on
    another thread while the sink is replaced may still reach the old one.
    Exceptions the sink raises are logged at debug level and otherwise ignored.

    Args:
        sink: The function to call with each usage event, or None to stop
            logging.
    """
    global _sink  # noqa: PLW0603
    _sink = sink


def log_usage(
    kind: EventKind,
    *,
    batched: bool = False,
    main_process_only: bool = False,
) -> Callable[
    [Callable[Concatenate[_EstimatorT, _P], _R]],
    Callable[Concatenate[_EstimatorT, _P], _R],
]:
    """Log each use of an estimator method as a usage event.

    A call is logged when it returns or raises, on the thread that made it,
    unless it was made by another logged call. The size of its data is read
    from its `X` argument; the data itself is never read.

    Args:
        kind: The kind of usage a call is logged as.
        batched: The method fits and predicts each dataset in `X_train_list` and
            `X_test_list`, and is logged as one prediction covering them all.
        main_process_only: Log calls only on the main process of a torchrun job,
            for methods that every process of the job runs together.

    Returns:
        A decorator for estimator methods.
    """

    def decorator(
        fn: Callable[Concatenate[_EstimatorT, _P], _R],
    ) -> Callable[Concatenate[_EstimatorT, _P], _R]:
        signature = inspect.signature(fn)
        _check_loggable(fn, signature, batched=batched)
        # The class that defines the method, and its name, such as
        # "TabPFNClassifier.predict". Just the name where the API would reject
        # that, as for a class defined inside a function.
        method = label_of(fn.__qualname__) or fn.__name__

        @functools.wraps(fn)
        def wrapper(
            estimator: _EstimatorT, /, *args: _P.args, **kwargs: _P.kwargs
        ) -> _R:
            if _in_logged_call.get():
                return fn(estimator, *args, **kwargs)

            token = _in_logged_call.set(True)
            started_at, started = datetime.now(timezone.utc), time.perf_counter()
            succeeded = False
            try:
                result = fn(estimator, *args, **kwargs)
                succeeded = True
                return result
            finally:
                _in_logged_call.reset(token)
                # Built by `_emit`, only when there is a sink, and where a failure
                # cannot reach the caller.
                builder = lambda: usage_event(
                    estimator,
                    _arguments(signature, estimator, args, kwargs),
                    kind=kind,
                    batched=batched,
                    method=method,
                    succeeded=succeeded,
                    started_at=started_at,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
                _emit(builder, main_process_only=main_process_only)

        return wrapper

    return decorator


def _emit(
    build_event: Callable[[], dict[str, Any] | None],
    *,
    main_process_only: bool,
) -> None:
    """Hand a finished call's usage event to the sink, if there is one.

    A failure only loses the call's event, never its result or exception.
    """
    # Read once, so the sink checked is the sink called.
    sink = _sink

    # torchrun sets RANK on every process of a job; 0 is the main one.
    if sink is None or (main_process_only and os.environ.get("RANK", "0") != "0"):
        return

    try:
        event = build_event()
        if event is not None:
            # Handed over as if still inside the call, so a logged method that
            # the sink calls counts as nested and is not logged again.
            context = copy_context()
            context.run(_in_logged_call.set, True)  # noqa: FBT003
            context.run(sink, event)
    except Exception:
        logger.debug("Could not log a call's usage.", exc_info=True)


def _arguments(
    signature: inspect.Signature,
    estimator: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Mapping[str, Any]:
    """The call's arguments by name, with the defaults of those not passed.

    Arguments collected by `**kwargs` are included under their own names. A
    call whose arguments do not match the signature has none, as the call
    itself fails on them.
    """
    try:
        bound = signature.bind_partial(estimator, *args, **kwargs)
    except TypeError:
        return {}
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    for name, parameter in signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            arguments.update(arguments.pop(name))
    return arguments


def _check_loggable(
    fn: Callable[..., Any], signature: inspect.Signature, *, batched: bool
) -> None:
    """Raise if `log_usage` cannot log the calls to `fn`."""
    if (
        inspect.isgeneratorfunction(fn)
        or inspect.iscoroutinefunction(fn)
        or inspect.isasyncgenfunction(fn)
    ):
        # The context variable would stay set across every yield or await.
        raise TypeError(f"log_usage() can only decorate plain functions, not {fn!r}.")
    data_params = ("X_train_list", "X_test_list") if batched else ("X",)
    missing = [name for name in data_params if name not in signature.parameters]
    if missing:
        raise TypeError(f"log_usage() reads the data from {missing}, not in {fn!r}.")
