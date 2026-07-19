"""The cooperative scheduler that serializes controlled threads.

Exactly one controlled thread runs at any moment.  Every bytecode
instruction (or line) of traced user code, and every modelled
synchronization operation, is a *scheduling point*: the running thread
stops, a strategy picks which thread runs next, and control is handed
off directly thread-to-thread.  Given the same model and the same
sequence of choices an execution is fully deterministic — which is what
makes failures replayable.

Because modelled primitives never block their OS thread (a logically
blocked thread parks inside the scheduler instead), the scheduler always
knows the exact set of runnable threads.  That is what enables both
systematic exploration and deadlock detection.
"""

from __future__ import annotations

import os
import sys
import sysconfig
import threading as _threading
import traceback as _traceback
from collections import deque
from contextlib import contextmanager
from enum import Enum
from time import monotonic as _monotonic
from typing import Any, Callable, Dict, List, Optional

import _thread

from ._records import Step

# Real primitives, captured before any monkeypatching can occur.  OS threads
# are spawned via the raw _thread module: threading.Thread.__init__ looks up
# Event/Condition in threading's module globals at call time, which would hand
# our own machinery the *modelled* primitives while the patch is active.
_start_new_thread = _thread.start_new_thread
_allocate_lock = _thread.allocate_lock
_RealLock = _threading.Lock
_RealCondition = _threading.Condition
_get_ident = _threading.get_ident

# Scheduling points in plain user code come from sys.monitoring (PEP 669),
# not sys.settrace: the legacy settrace path applies per-code-object
# instrumentation lazily, so the first-ever execution of a function delivers
# fewer events than later ones — fatal for deterministic replay and DFS
# revisits.  sys.monitoring delivers identical events from the first
# execution on (verified empirically; see README "How it works").
_MON = sys.monitoring
_TOOL_NAME = "interleave-test"

_PKG_DIR = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))

# The stdlib directory is derived from threading.__file__ (not only
# sysconfig): sysconfig may report paths through a symlink (e.g. Homebrew's
# /opt/homebrew/opt/...) while code objects carry the resolved path, and a
# missed match here is fatal — tracing threading.py means tracing the very
# machinery the scheduler runs on.
_STDLIB_DIR = os.path.realpath(os.path.dirname(_threading.__file__))

_EXCLUDED_DIRS = []
for _key in ("stdlib", "platstdlib", "purelib", "platlib"):
    _p = sysconfig.get_paths().get(_key)
    if _p:
        _p = os.path.realpath(_p)
        if _p not in _EXCLUDED_DIRS:
            _EXCLUDED_DIRS.append(_p)


class _Abort(BaseException):
    """Raised inside controlled threads to unwind them during teardown.

    Derives from BaseException so ordinary ``except Exception`` handlers
    in user code do not swallow it.
    """


def acquire_tool_id() -> int:
    """Claim a free sys.monitoring tool id for the duration of a run."""
    for tool in (3, 4, 2, 1, 5, 0):
        try:
            _MON.use_tool_id(tool, _TOOL_NAME)
            return tool
        except ValueError:
            continue
    raise RuntimeError(
        "no free sys.monitoring tool id: another tracing tool (debugger, "
        "coverage, profiler) has claimed all slots"
    )


def _line_for_offset(code, offset: int) -> int:
    for start, end, line in code.co_lines():
        if line is not None and start <= offset < end:
            return line
    return code.co_firstlineno


def _fmt_loc(loc) -> Optional[str]:
    """Resolve a lazily recorded location to 'file.py:lineno'."""
    if loc is None or isinstance(loc, str):
        return loc
    kind, code, value = loc
    name = os.path.basename(code.co_filename)
    if kind == "line":
        return "%s:%d" % (name, value)
    return "%s:%d" % (name, _line_for_offset(code, value))


class Status(Enum):
    READY = "ready"  # wants to run, waiting to be granted
    RUNNING = "running"  # the single thread currently executing
    BLOCKED = "blocked"  # logically blocked on a WaitSet
    FINISHED = "finished"


class WaitSet:
    """A queue of threads logically blocked on the same thing."""

    __slots__ = ("waiters", "describe")

    def __init__(self, describe: Any = "?"):
        self.waiters: List["TRec"] = []
        self.describe = describe  # str or zero-arg callable, for diagnostics

    def describe_now(self) -> str:
        d = self.describe
        return d() if callable(d) else str(d)


class TRec:
    """Scheduler bookkeeping for one controlled thread."""

    __slots__ = (
        "tid",
        "name",
        "status",
        "granted",
        "woke_by_timeout",
        "blocked_on",
        "deadline",
        "done",
        "result",
        "join_ws",
        "last_location",
        "no_yield",
    )

    def __init__(self, tid: int, name: str):
        self.tid = tid
        self.name = name
        self.status = Status.READY
        self.granted = False
        self.woke_by_timeout = False
        self.blocked_on: Optional[WaitSet] = None
        self.deadline: Optional[float] = None  # virtual-clock deadline
        self.done = _allocate_lock()  # held while the OS thread is alive
        self.result: Any = None
        self.join_ws = WaitSet(lambda: "join(T%d(%s))" % (self.tid, self.name))
        self.last_location: Optional[str] = None
        self.no_yield = 0


class _FailureInfo:
    __slots__ = ("kind", "exc", "tb", "details")

    def __init__(self, kind: str, exc: Optional[BaseException], tb: Optional[str], details: str):
        self.kind = kind
        self.exc = exc
        self.tb = tb
        self.details = details


# The scheduler of the run currently in progress (at most one per process).
_active: Optional["Scheduler"] = None


def current_scheduler() -> Optional["Scheduler"]:
    return _active


class Task:
    """Handle to a controlled thread, returned by ``spawn()``."""

    def __init__(self, sched: "Scheduler", rec: TRec):
        self._sched = sched
        self._rec = rec

    @property
    def name(self) -> str:
        return self._rec.name

    def is_alive(self) -> bool:
        return self._rec.status is not Status.FINISHED

    def join(self, timeout: Optional[float] = None) -> None:
        """Logically wait for this thread to finish (a scheduling point)."""
        sched, rec = self._sched, self._rec
        me = sched.me()
        if me is None:
            raise RuntimeError(
                "Task.join() must be called from inside the controlled run"
            )
        sched.yield_point("join")
        with sched._cv:
            if sched._abort:
                raise _Abort()
            if rec.status is Status.FINISHED:
                return
            if timeout is not None and timeout <= 0:
                return  # an expired timeout is a poll, not a park
            sched._handoff(
                me,
                rec.join_ws,
                float(timeout) if timeout is not None else None,
                "join",
            )

    def result(self) -> Any:
        """Return value of the thread's target (join() it first)."""
        if self._rec.status is not Status.FINISHED:
            raise RuntimeError("thread has not finished; join() it first")
        return self._rec.result


class Scheduler:
    """Serializes a set of controlled threads and records every decision."""

    def __init__(
        self,
        strategy,
        *,
        tool_id: int,
        granularity: str = "opcode",
        include=None,
        exclude=None,
        record_steps: bool = True,
    ):
        self._strategy = strategy
        self._tool_id = tool_id
        self._granularity = granularity
        self._include = [os.path.realpath(p) for p in include] if include else None
        self._exclude = [os.path.realpath(p) for p in exclude] if exclude else []
        self._cv = _RealCondition(_RealLock())
        self._recs: List[TRec] = []
        self._byid: Dict[int, TRec] = {}
        self._fcache: Dict[str, bool] = {}
        self._decisions: List[int] = []
        self._steps = deque(maxlen=1000) if record_steps else None
        self._abort = False
        self._done = False
        self._failure: Optional[_FailureInfo] = None
        self._leaked = 0
        self._vnow = 0.0  # virtual clock; advanced by logical timeout firing
        # Optional strategy hook for forced (single-runnable) decisions.
        self._observe = getattr(strategy, "observe", None)

    # ------------------------------------------------------------------ #
    # thread identity                                                    #
    # ------------------------------------------------------------------ #

    def me(self) -> Optional[TRec]:
        return self._byid.get(_get_ident())

    # ------------------------------------------------------------------ #
    # spawning                                                           #
    # ------------------------------------------------------------------ #

    def spawn(self, fn: Callable, *args, name: Optional[str] = None, **kwargs) -> Task:
        caller = self.me()
        with self._cv:
            if self._abort:
                if caller is not None:
                    raise _Abort()
                raise RuntimeError("cannot spawn: the run is aborting")
            tid = len(self._recs)
            rec = TRec(tid, name or getattr(fn, "__name__", "thread"))
            self._recs.append(rec)
            rec.done.acquire()  # released by _thread_main on the way out
        _start_new_thread(self._thread_main, (rec, fn, args, kwargs))
        if caller is not None:
            self.yield_point("spawn")
        return Task(self, rec)

    def _thread_main(self, rec: TRec, fn: Callable, args, kwargs) -> None:
        try:
            with self._cv:
                self._byid[_get_ident()] = rec
                aborted = False
                while True:
                    if self._abort:
                        aborted = True
                        break
                    if rec.granted:
                        rec.granted = False
                        rec.status = Status.RUNNING
                        break
                    self._cv.wait()
                if aborted:
                    self._finish_thread(rec)
                    return
            try:
                rec.result = fn(*args, **kwargs)
            except _Abort:
                pass
            except BaseException as exc:
                with self._cv:
                    self._fail(
                        "exception", exc, "raised in T%d(%s)" % (rec.tid, rec.name)
                    )
            finally:
                with self._cv:
                    self._finish_thread(rec)
        finally:
            rec.done.release()

    # ------------------------------------------------------------------ #
    # tracing (scheduling points in plain user code)                     #
    # ------------------------------------------------------------------ #

    def _should_trace(self, filename: str) -> bool:
        cached = self._fcache.get(filename)
        if cached is not None:
            return cached
        result = self._compute_should_trace(filename)
        self._fcache[filename] = result
        return result

    def _compute_should_trace(self, filename: str) -> bool:
        if not filename or filename.startswith("<"):
            return False
        path = os.path.realpath(os.path.abspath(filename))
        sep = os.sep
        # Unconditional: interleave_test itself and the stdlib (which
        # includes threading.py, the machinery the scheduler runs on).
        if path.startswith(_PKG_DIR + sep):
            return False
        if path.startswith(_STDLIB_DIR + sep):
            return False
        # User-supplied exclude always wins, including inside include=.
        for p in self._exclude:
            if path.startswith(p + sep) or path == p:
                return False
        if self._include is not None:
            return any(
                path.startswith(p + sep) or path == p for p in self._include
            )
        for p in _EXCLUDED_DIRS:
            if path.startswith(p + sep):
                return False
        return True

    def _on_instruction(self, code, offset):
        # sys.monitoring callback: fires in every thread; filter here.
        if not self._should_trace(code.co_filename):
            return _MON.DISABLE  # path-based: safe to disable the location
        rec = self._byid.get(_get_ident())
        if rec is None:
            return None  # untracked thread: keep the event enabled
        if self._abort:
            # Checked before no_yield so threads spinning inside a
            # no_interleave() region can still be torn down.
            raise _Abort()
        if rec.no_yield:
            return None
        rec.last_location = ("i", code, offset)
        with self._cv:
            if self._abort:
                raise _Abort()
            self._handoff(rec, None, None, "step")
        return None

    def _on_line(self, code, lineno):
        if not self._should_trace(code.co_filename):
            return _MON.DISABLE
        rec = self._byid.get(_get_ident())
        if rec is None:
            return None
        if self._abort:
            raise _Abort()
        if rec.no_yield:
            return None
        rec.last_location = ("line", code, lineno)
        with self._cv:
            if self._abort:
                raise _Abort()
            self._handoff(rec, None, None, "step")
        return None

    def _events_on(self) -> None:
        if self._granularity == "opcode":
            _MON.register_callback(
                self._tool_id, _MON.events.INSTRUCTION, self._on_instruction
            )
            _MON.set_events(self._tool_id, _MON.events.INSTRUCTION)
        else:
            _MON.register_callback(self._tool_id, _MON.events.LINE, self._on_line)
            _MON.set_events(self._tool_id, _MON.events.LINE)

    def _events_off(self) -> None:
        _MON.set_events(self._tool_id, 0)

    # ------------------------------------------------------------------ #
    # scheduling points                                                  #
    # ------------------------------------------------------------------ #

    def yield_point(self, kind: str = "step") -> None:
        """A voluntary scheduling point (used by modelled primitives)."""
        rec = self._byid.get(_get_ident())
        if rec is None:
            return
        if self._abort:
            raise _Abort()
        if rec.no_yield:
            return
        rec.last_location = self._user_location()
        with self._cv:
            if self._abort:
                raise _Abort()
            self._handoff(rec, None, None, kind)

    def _user_location(self) -> str:
        f = sys._getframe(2)
        for _ in range(30):
            if f is None:
                break
            filename = f.f_code.co_filename
            if self._should_trace(filename):
                return "%s:%d" % (os.path.basename(filename), f.f_lineno)
            f = f.f_back
        return "?"

    def _handoff(
        self,
        rec: TRec,
        waitable: Optional[WaitSet],
        timeout: Optional[float],
        kind: str,
    ) -> bool:
        """Hand control to the next thread.  Called under ``self._cv`` by
        the currently running thread.  Returns True if a logical timeout
        woke us rather than the event we were waiting for."""
        if self._abort:
            raise _Abort()
        if waitable is None:
            rec.status = Status.READY
            current: Optional[TRec] = rec
        else:
            rec.status = Status.BLOCKED
            rec.blocked_on = waitable
            rec.deadline = None if timeout is None else self._vnow + timeout
            waitable.waiters.append(rec)
            current = None
        nxt = self._pick_next(current, kind)
        if nxt is rec:
            # A blocked rec can be rechosen immediately when its own logical
            # timeout fired; the flag must be consumed here too.
            rec.status = Status.RUNNING
            timed_out = rec.woke_by_timeout
            rec.woke_by_timeout = False
            return timed_out
        if nxt is not None:
            nxt.granted = True
            self._cv.notify_all()
        while True:
            if self._abort:
                rec.granted = False
                # Leave no trace in any WaitSet: a rec unwinding toward
                # FINISHED must not be resurrected by later wake-ups.
                if rec.blocked_on is not None:
                    try:
                        rec.blocked_on.waiters.remove(rec)
                    except ValueError:
                        pass
                    rec.blocked_on = None
                    rec.deadline = None
                raise _Abort()
            if rec.granted:
                break
            self._cv.wait()
        rec.granted = False
        rec.status = Status.RUNNING
        timed_out = rec.woke_by_timeout
        rec.woke_by_timeout = False
        return timed_out

    def _pick_next(self, current: Optional[TRec], kind: str) -> Optional[TRec]:
        """Choose the next thread to run.  Called under ``self._cv``."""
        while True:
            runnable = [r for r in self._recs if r.status is Status.READY]
            if runnable:
                if len(runnable) == 1:
                    chosen = runnable[0]
                    if self._observe is not None:
                        # Same reentrancy protection as strategy.pick below.
                        me = self.me()
                        if me is not None:
                            me.no_yield += 1
                        try:
                            self._observe(chosen, len(self._decisions))
                        finally:
                            if me is not None:
                                me.no_yield -= 1
                else:
                    # User strategies are traced user code, and we hold the
                    # scheduler cv here: a scheduling point inside pick()
                    # would self-deadlock.  Suppress yields for the call.
                    me = self.me()
                    if me is not None:
                        me.no_yield += 1
                    try:
                        chosen = self._strategy.pick(
                            runnable, current, len(self._decisions)
                        )
                    finally:
                        if me is not None:
                            me.no_yield -= 1
                    if chosen not in runnable:
                        raise RuntimeError("strategy picked a non-runnable thread")
                self._decisions.append(chosen.tid)
                if self._steps is not None:
                    self._steps.append(
                        Step(
                            index=len(self._decisions) - 1,
                            chosen=chosen.tid,
                            runnable=tuple(r.tid for r in runnable),
                            kind=kind,
                            location=_fmt_loc(chosen.last_location) or "(start)",
                            thread_name=chosen.name,
                        )
                    )
                return chosen
            live = [r for r in self._recs if r.status is Status.BLOCKED]
            if not live:
                self._done = True
                self._cv.notify_all()
                return None
            timed = [r for r in live if r.deadline is not None]
            if timed:
                # Nothing can run: the earliest logical timeout fires now,
                # advancing the virtual clock to its deadline.
                r = min(timed, key=lambda x: (x.deadline, x.tid))
                self._vnow = max(self._vnow, r.deadline)
                r.blocked_on.waiters.remove(r)
                r.blocked_on = None
                r.deadline = None
                r.woke_by_timeout = True
                r.status = Status.READY
                continue
            self._fail("deadlock", None, self._blocked_table(live))
            return None

    def _blocked_table(self, live: List[TRec]) -> str:
        # describe_now() may execute user callables while self._cv is held;
        # suppress scheduling points for the duration.
        me = self.me()
        if me is not None:
            me.no_yield += 1
        try:
            return self._blocked_table_inner(live)
        finally:
            if me is not None:
                me.no_yield -= 1

    def _blocked_table_inner(self, live: List[TRec]) -> str:
        lines = []
        for r in live:
            what = r.blocked_on.describe_now() if r.blocked_on else "?"
            lines.append(
                "  T%d(%s)  blocked on %s  at %s"
                % (r.tid, r.name, what, _fmt_loc(r.last_location) or "?")
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # wake-ups (called under self._cv by modelled primitives)            #
    # ------------------------------------------------------------------ #

    def _wake_all(self, ws: WaitSet) -> None:
        for r in ws.waiters:
            r.blocked_on = None
            r.deadline = None
            # Never resurrect a thread that already finished (it can be
            # left in a WaitSet only during abort teardown).
            if r.status is not Status.FINISHED:
                r.status = Status.READY
        ws.waiters.clear()

    def _wake_n(self, ws: WaitSet, n: int) -> None:
        woken = 0
        while woken < n and ws.waiters:
            r = ws.waiters.pop(0)
            r.blocked_on = None
            r.deadline = None
            if r.status is not Status.FINISHED:
                r.status = Status.READY
                woken += 1

    # ------------------------------------------------------------------ #
    # completion, failure, teardown                                      #
    # ------------------------------------------------------------------ #

    def _finish_thread(self, rec: TRec) -> None:
        """Mark ``rec`` finished and hand off / signal completion.
        Called under ``self._cv``."""
        rec.status = Status.FINISHED
        self._wake_all(rec.join_ws)
        if self._abort:
            if all(r.status is Status.FINISHED for r in self._recs):
                self._done = True
            self._cv.notify_all()
            return
        try:
            nxt = self._pick_next(None, "exit")
        except BaseException as exc:
            # A strategy exception at the exit decision must reach the
            # failure machinery, not escape into a raw OS thread.
            self._fail(
                "exception",
                exc,
                "raised at the exit decision of T%d(%s)" % (rec.tid, rec.name),
            )
            return
        if nxt is not None:
            nxt.granted = True
            self._cv.notify_all()

    def _fail(self, kind: str, exc: Optional[BaseException], details: str = "") -> None:
        """Record the first failure and begin aborting.  Under ``self._cv``."""
        if self._failure is None:
            tb = None
            if exc is not None:
                # format_exception() runs str(exc): a user-defined __str__
                # lives in traced code, and we hold self._cv — a scheduling
                # point here would self-deadlock the process.
                me = self.me()
                if me is not None:
                    me.no_yield += 1
                try:
                    tb = "".join(
                        _traceback.format_exception(
                            type(exc), exc, exc.__traceback__
                        )
                    )
                finally:
                    if me is not None:
                        me.no_yield -= 1
            self._failure = _FailureInfo(kind, exc, tb, details)
        self._abort = True
        self._cv.notify_all()

    def report_external_error(self, exc: BaseException) -> None:
        """Fail the schedule from a thread the scheduler does not manage.

        Used by modelled primitives when touched from an unmanaged OS
        thread: raising there alone would be swallowed by the raw thread
        and leave the run to die as a 30s hang.
        """
        with self._cv:
            self._fail("exception", exc, "raised on an unmanaged thread")

    def run(self, timeout: float) -> Optional[_FailureInfo]:
        """Grant the first thread and wait for the schedule to complete.

        Returns the failure info, or None if the schedule ran clean.
        """
        deadline = _monotonic() + timeout
        self._events_on()
        try:
            with self._cv:
                nxt = self._pick_next(None, "start")
                if nxt is not None:
                    nxt.granted = True
                    self._cv.notify_all()
                while not self._done and self._failure is None:
                    remaining = deadline - _monotonic()
                    if remaining <= 0:
                        self._fail(
                            "hang",
                            None,
                            "the schedule exceeded the %.1fs per-schedule timeout; "
                            "the model is probably stuck in a real blocking call the "
                            "scheduler does not model (C-level I/O, an unpatched "
                            "primitive, a long computation, ...)" % timeout,
                        )
                        break
                    self._cv.wait(remaining)
                if self._failure is not None:
                    self._abort = True
                    self._cv.notify_all()
                    grace = _monotonic() + 5.0
                    while not self._done and _monotonic() < grace:
                        self._cv.wait(0.1)
            for rec in self._recs:
                if rec.done.acquire(True, 1.0):
                    rec.done.release()
                else:
                    self._leaked += 1
        finally:
            self._events_off()
        return self._failure


@contextmanager
def no_interleave():
    """Suppress scheduling points for the current thread inside the block.

    Everything inside the block executes atomically with respect to the
    other controlled threads.  Useful to make setup code atomic, or to
    assert that guarding a region does fix a race.  No-op outside a run.
    """
    sched = _active
    rec = sched.me() if sched is not None else None
    if rec is None:
        yield
        return
    rec.no_yield += 1
    try:
        yield
    finally:
        rec.no_yield -= 1
