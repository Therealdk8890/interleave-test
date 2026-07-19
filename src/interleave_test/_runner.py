"""Public entry points: explore(), replay(), spawn(), @interleave."""

from __future__ import annotations

import functools
import inspect
import os
import platform
import queue as _queue
import random as _random
import threading as _threading
import time as _time
import warnings
from contextlib import contextmanager
from time import monotonic as _monotonic
from typing import Callable, Optional, Sequence, Union

from . import _primitives as _p
from . import _scheduler as _s
from ._errors import DeadlockError, HangError, InterleaveError, RaceFound
from ._records import Failure, Result, Schedule
from ._strategies import (
    DFSStrategy,
    PCTStrategy,
    RandomStrategy,
    ReplayStrategy,
    Strategy,
)

# Captured before any patching (module import happens unpatched).
_run_lock = _threading.Lock()

# Filter configuration of the previous explore() in this process, used to
# decide whether sys.monitoring DISABLE state must be reset.  restart_events()
# is global across tools, so calling it unconditionally would clobber e.g.
# coverage's own DISABLE optimizations on every explore().
_last_filter_key = None


def _make_strategy(spec, max_preemptions: int) -> Strategy:
    if isinstance(spec, Strategy):
        return spec
    if spec == "random":
        return RandomStrategy()
    if spec == "pct":
        return PCTStrategy()
    if spec == "dfs":
        return DFSStrategy(max_preemptions=max_preemptions)
    raise ValueError(
        "unknown strategy %r; expected 'random', 'pct', 'dfs', or a Strategy "
        "instance" % (spec,)
    )


def _norm_paths(name: str, value) -> Optional[list]:
    """Normalize include/exclude to a list of realpath strings.

    Accepts a sequence of paths, or a single str/PathLike (wrapped) — but
    never iterates a bare string as characters.
    """
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        value = [value]
    out = []
    for item in value:
        if not isinstance(item, (str, os.PathLike)):
            raise TypeError(
                "%s must be path strings, got %s" % (name, type(item).__name__)
            )
        out.append(os.path.realpath(os.fspath(item)))
    return out


@contextmanager
def _patched(patch_modules: bool, patch_time: bool):
    targets = []
    if patch_modules:
        targets += [
            (_threading, "Lock", _p.Lock),
            (_threading, "RLock", _p.RLock),
            (_threading, "Condition", _p.Condition),
            (_threading, "Event", _p.Event),
            (_threading, "Semaphore", _p.Semaphore),
            (_threading, "BoundedSemaphore", _p.BoundedSemaphore),
            (_threading, "Thread", _p.Thread),
        ]
        # SimpleQueue is C-implemented: its get() would block the OS thread
        # invisibly.  The pure-Python fallback builds on (patched) threading.
        pysq = getattr(_queue, "_PySimpleQueue", None)
        if pysq is not None:
            targets.append((_queue, "SimpleQueue", pysq))
    if patch_time:
        targets += [
            (_time, "sleep", _p.sleep),
            (_time, "monotonic", _p.monotonic),
            (_time, "time", _p.time),
            # queue.py binds `from time import monotonic as time` at import.
            (_queue, "time", _p.monotonic),
        ]
    saved = []
    for mod, attr, repl in targets:
        saved.append((mod, attr, getattr(mod, attr)))
        setattr(mod, attr, repl)
    try:
        yield
    finally:
        for mod, attr, orig in reversed(saved):
            setattr(mod, attr, orig)


def explore(
    model: Callable[[], object],
    *,
    iterations: int = 100,
    seed: Optional[int] = None,
    strategy: Union[str, Strategy] = "random",
    granularity: str = "opcode",
    max_preemptions: int = 2,
    timeout: float = 30.0,
    include: Optional[Sequence] = None,
    exclude: Optional[Sequence] = None,
    patch_modules: bool = True,
    patch_time: bool = True,
    raise_on_failure: bool = True,
) -> Result:
    """Run ``model`` under many thread interleavings and report the first bad one.

    ``model`` is a zero-argument callable that builds its own state,
    starts threads (via :func:`interleave_test.spawn` or
    ``threading.Thread``), joins them, and asserts.  It runs once per
    schedule and must be deterministic apart from thread scheduling.

    Parameters:

    * ``iterations`` — maximum number of schedules to run (all strategies).
    * ``seed`` — root seed; the run is fully reproducible given the same
      seed on the same interpreter version.  Randomly drawn (and
      reported) when omitted.
    * ``strategy`` — ``"random"`` (uniform), ``"pct"`` (probabilistic
      concurrency testing; better for rare orderings), ``"dfs"`` (bounded
      exhaustive search), or a :class:`~interleave_test.Strategy`.
    * ``granularity`` — ``"opcode"`` interleaves between bytecode
      instructions (catches ``x += 1`` races; default), ``"line"`` is
      faster but line-atomic.
    * ``max_preemptions`` — preemption bound for ``"dfs"``.
    * ``timeout`` — wall-clock seconds allowed per schedule before the
      run is reported as a hang.
    * ``include`` / ``exclude`` — path-prefix sequences controlling which
      files are traced (a single path is accepted too).  By default all
      user code is traced; the standard library, site-packages, and
      interleave_test itself are not.  ``exclude`` always wins.
    * ``patch_modules`` / ``patch_time`` — patch ``threading`` names,
      ``queue.SimpleQueue``, and ``time.sleep``/``monotonic``/``time``
      inside the run (on by default).
    * ``raise_on_failure`` — raise :class:`RaceFound` /
      :class:`DeadlockError` / :class:`HangError` on the first failing
      schedule (default), or return a :class:`Result` describing it.
    """
    if granularity not in ("opcode", "line"):
        raise ValueError("granularity must be 'opcode' or 'line'")
    if iterations < 0:
        raise ValueError("iterations must be >= 0")
    include = _norm_paths("include", include)
    exclude = _norm_paths("exclude", exclude)
    strat = _make_strategy(strategy, max_preemptions)
    if iterations > 0 and strat.more() is False:
        raise RuntimeError(
            "the provided strategy has no schedules left to run (a "
            "DFSStrategy instance is single-use once exhausted)"
        )
    replaying = isinstance(strat, ReplayStrategy)
    if replaying:
        report_seed = seed  # replay has no seed of its own
        derive_base = 0
    else:
        if seed is None:
            seed = _random.SystemRandom().randrange(2**32)
        report_seed = seed
        derive_base = seed
    if not _run_lock.acquire(blocking=False):
        raise RuntimeError(
            "explore() is already running in this process; nested or "
            "concurrent explorations are not supported"
        )
    global _last_filter_key
    start_time = _monotonic()
    failure: Optional[Failure] = None
    leaked = 0
    i = 0
    try:
        tool_id = _s.acquire_tool_id()
        try:
            filter_key = (
                tool_id,
                tuple(include) if include is not None else None,
                tuple(exclude) if exclude is not None else (),
            )
            if _last_filter_key is not None and _last_filter_key != filter_key:
                # Our per-location DISABLEs from a previous run used a
                # different filter; they must be cleared.  (restart_events
                # is global, so we avoid it when nothing changed.)
                _s._MON.restart_events()
            _last_filter_key = filter_key
            with _patched(patch_modules, patch_time):
                while i < iterations and strat.more() is not False:
                    strat.begin(i, derive_base * 1_000_003 + i)
                    sched = _s.Scheduler(
                        strat,
                        tool_id=tool_id,
                        granularity=granularity,
                        include=include,
                        exclude=exclude,
                    )
                    _s._active = sched
                    try:
                        sched.spawn(model, name=getattr(model, "__name__", "model"))
                        info = sched.run(timeout)
                    finally:
                        _s._active = None
                    strat.end()
                    leaked += sched._leaked
                    i += 1
                    if info is not None:
                        failure = Failure(
                            kind=info.kind,
                            iteration=i - 1,
                            seed=report_seed,
                            schedule=Schedule(
                                choices=list(sched._decisions),
                                granularity=granularity,
                                python=platform.python_version(),
                            ),
                            steps=list(sched._steps or ()),
                            exception=info.exc,
                            traceback=info.tb,
                            details=info.details,
                            leaked_threads=sched._leaked,
                        )
                        break
        finally:
            _s._MON.free_tool_id(tool_id)
    finally:
        _run_lock.release()
    result = Result(
        schedules_run=i,
        seed=report_seed,
        failure=failure,
        exhausted=strat.exhausted,
        elapsed=_monotonic() - start_time,
        leaked_threads=leaked,
    )
    if failure is None and leaked:
        warnings.warn(
            "%d controlled thread(s) could not be terminated and were leaked; "
            "later work in this process may be affected" % leaked,
            RuntimeWarning,
            stacklevel=2,
        )
    if failure is not None and raise_on_failure:
        if isinstance(failure.exception, InterleaveError):
            # e.g. ReplayDivergence raised inside the run: surface as itself.
            failure.exception.failure = failure
            failure.exception.result = result
            raise failure.exception
        cls = {
            "exception": RaceFound,
            "deadlock": DeadlockError,
            "hang": HangError,
        }[failure.kind]
        err = cls(failure.format())
        err.failure = failure
        err.result = result
        raise err from failure.exception
    return result


def replay(
    model: Callable[[], object],
    schedule,
    *,
    timeout: float = 30.0,
    include: Optional[Sequence] = None,
    exclude: Optional[Sequence] = None,
    patch_modules: bool = True,
    patch_time: bool = True,
    raise_on_failure: bool = True,
) -> Result:
    """Re-run ``model`` once, following a recorded schedule exactly.

    ``schedule`` may be a :class:`~interleave_test.Schedule`, a
    :class:`~interleave_test.Failure`, or a JSON string produced by
    :meth:`Schedule.to_json`.  The recording's granularity is always
    used; schedules are interpreter-version-specific (they count
    bytecode instructions), so replay on the Python version that
    recorded them.
    """
    orig_seed = None
    if isinstance(schedule, Failure):
        orig_seed = schedule.seed
        schedule = schedule.schedule
    if isinstance(schedule, str):
        schedule = Schedule.from_json(schedule)
    if schedule.python and schedule.python != platform.python_version():
        warnings.warn(
            "schedule was recorded on Python %s but this is %s; schedules "
            "count bytecode instructions and are interpreter-version-"
            "specific — expect ReplayDivergence if the bytecode differs"
            % (schedule.python, platform.python_version()),
            RuntimeWarning,
            stacklevel=2,
        )
    return explore(
        model,
        iterations=1,
        seed=orig_seed,
        strategy=ReplayStrategy(schedule),
        granularity=schedule.granularity,
        timeout=timeout,
        include=include,
        exclude=exclude,
        patch_modules=patch_modules,
        patch_time=patch_time,
        raise_on_failure=raise_on_failure,
    )


def spawn(fn: Callable, *args, name: Optional[str] = None, **kwargs) -> _s.Task:
    """Start a controlled thread running ``fn(*args, **kwargs)``.

    Only valid inside a model running under :func:`explore` /
    :func:`replay`.  Returns a :class:`~interleave_test.Task` with
    ``join()`` and ``result()``.
    """
    sched = _s.current_scheduler()
    if sched is None or sched.me() is None:
        raise RuntimeError(
            "spawn() may only be called from inside a model running under "
            "explore()/replay()"
        )
    return sched.spawn(fn, *args, name=name, **kwargs)


def interleave(model: Optional[Callable] = None, **options):
    """Decorator form of :func:`explore`.

    ::

        @interleave(iterations=500, seed=0)
        def test_counter():
            ...model body using spawn()...

    Calling the decorated function runs the exploration, so a bare
    ``pytest`` picks it up like any other test.  The wrapper returns
    None (pytest warns on tests returning values); the last
    :class:`~interleave_test.Result` is available as
    ``test_counter.last_result`` and the undecorated model as
    ``test_counter.model`` (handy for ``replay``).

    Models take no parameters — pytest fixtures are not supported; use
    closure state instead.
    """

    def decorate(fn: Callable):
        for param in inspect.signature(fn).parameters.values():
            if param.default is inspect.Parameter.empty and param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                raise TypeError(
                    "@interleave model %r must take no required parameters; "
                    "models are called with no arguments — use closure state "
                    "instead (pytest fixtures are not supported)"
                    % fn.__name__
                )

        @functools.wraps(fn)
        def wrapper():
            wrapper.last_result = explore(fn, **options)
            return None

        # functools.wraps copies __wrapped__, which would make pytest see
        # the model's signature and try to inject fixtures; the explicit
        # empty signature wins.
        wrapper.__signature__ = inspect.Signature()
        wrapper.model = fn
        wrapper.last_result = None
        return wrapper

    if model is not None:
        return decorate(model)
    return decorate
