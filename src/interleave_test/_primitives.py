"""Scheduler-aware drop-in replacements for ``threading`` primitives.

These model the semantics of the real primitives inside the scheduler's
state machine, so a logically blocked thread never blocks its OS thread.
The scheduler therefore always knows exactly which threads can run —
the property that enables deterministic exploration, replay, and
deadlock detection.

During :func:`interleave_test.explore` the names ``threading.Lock``,
``RLock``, ``Condition``, ``Event``, ``Semaphore``, ``BoundedSemaphore``
and ``Thread`` (plus ``time.sleep``, ``time.monotonic``, ``time.time``
and ``queue.SimpleQueue``) are patched to these implementations, so code
that creates its synchronization objects inside the model — which
includes ``queue.Queue`` — is modelled automatically.  Note the patch is
applied to the modules themselves, so ``threading.py``'s own globals are
patched too: a real ``threading.Thread`` class captured *before* the run
and instantiated *inside* it would mix modelled and real machinery —
touching a modelled primitive from such an unmanaged thread fails the
schedule immediately with a descriptive error.

Timeouts are *logical*: a timed wait expires only when no thread can
run, at which point the scheduler's virtual clock (served by the patched
``time.monotonic``/``time.time``) jumps to the deadline.
"""

from __future__ import annotations

import time as _time
from typing import Any, Callable, Optional

from ._scheduler import WaitSet, current_scheduler

_real_sleep = _time.sleep
_real_monotonic = _time.monotonic
_real_time = _time.time

# Fixed epoch for the virtual time.time(): determinism requires a constant.
_TIME_EPOCH = 1_600_000_000.0


def _ctx(opname: str):
    sched = current_scheduler()
    if sched is None:
        raise RuntimeError(
            "%s: no interleave_test run is active; interleave_test primitives "
            "only work inside explore()/replay()" % opname
        )
    rec = sched.me()
    if rec is None:
        exc = RuntimeError(
            "%s called from a thread not managed by interleave_test; this "
            "usually means a threading class captured before explore() was "
            "instantiated inside the run, or a raw/foreign OS thread touched "
            "a modelled primitive" % opname
        )
        # Raising alone would be swallowed by a raw OS thread and the run
        # would die as a hang; report it to the scheduler first.
        sched.report_external_error(exc)
        raise exc
    return sched, rec


def _tmo(timeout) -> Optional[float]:
    """Normalize a threading-style timeout (-1/None = wait forever)."""
    if timeout is None or timeout < 0:
        return None
    return float(timeout)


class Lock:
    """Modelled ``threading.Lock``."""

    def __init__(self):
        self._owner = None  # TRec | None
        self._ws = WaitSet(self._describe)

    def _describe(self) -> str:
        o = self._owner
        if o is None:
            return "Lock (unlocked)"
        return "Lock held by T%d(%s)" % (o.tid, o.name)

    def acquire(self, blocking: bool = True, timeout=-1) -> bool:
        sched, rec = _ctx("Lock.acquire")
        if not blocking and timeout != -1:
            raise ValueError("can't specify a timeout for a non-blocking call")
        if timeout is not None and timeout < 0 and timeout != -1:
            raise ValueError("timeout value must be a non-negative number")
        sched.yield_point("lock.acquire")
        tmo = _tmo(timeout)
        with sched._cv:
            while self._owner is not None:
                if not blocking or tmo == 0.0:
                    return False  # an expired timeout is a poll, not a park
                if sched._handoff(rec, self._ws, tmo, "lock.blocked"):
                    return False  # logical timeout fired
            self._owner = rec
        return True

    def release(self) -> None:
        sched, rec = _ctx("Lock.release")
        with sched._cv:
            if self._owner is None:
                if sched._abort:
                    return  # forgiving during teardown unwinds
                raise RuntimeError("release unlocked lock")
            self._owner = None
            sched._wake_all(self._ws)
        sched.yield_point("lock.release")

    def locked(self) -> bool:
        return self._owner is not None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    # -- Condition integration ----------------------------------------- #

    def _is_owned_by(self, rec) -> bool:
        return self._owner is rec

    def _cond_release(self, sched):
        # Under sched._cv, without a scheduling point: Condition.wait must
        # release the lock and park atomically.
        self._owner = None
        sched._wake_all(self._ws)
        return None

    def _cond_reacquire(self, state) -> None:
        self.acquire()


class RLock:
    """Modelled ``threading.RLock``."""

    def __init__(self):
        self._owner = None
        self._count = 0
        self._ws = WaitSet(self._describe)

    def _describe(self) -> str:
        o = self._owner
        if o is None:
            return "RLock (unlocked)"
        return "RLock held by T%d(%s) (count=%d)" % (o.tid, o.name, self._count)

    def acquire(self, blocking: bool = True, timeout=-1) -> bool:
        sched, rec = _ctx("RLock.acquire")
        if not blocking and timeout != -1:
            raise ValueError("can't specify a timeout for a non-blocking call")
        if timeout is not None and timeout < 0 and timeout != -1:
            raise ValueError("timeout value must be a non-negative number")
        if self._owner is rec:
            self._count += 1
            return True
        sched.yield_point("lock.acquire")
        tmo = _tmo(timeout)
        with sched._cv:
            while self._owner is not None:
                if not blocking or tmo == 0.0:
                    return False
                if sched._handoff(rec, self._ws, tmo, "lock.blocked"):
                    return False
            self._owner = rec
            self._count = 1
        return True

    def release(self) -> None:
        sched, rec = _ctx("RLock.release")
        if self._owner is not rec:
            if sched._abort:
                return
            raise RuntimeError("cannot release un-acquired RLock")
        self._count -= 1
        if self._count == 0:
            with sched._cv:
                self._owner = None
                sched._wake_all(self._ws)
            sched.yield_point("lock.release")

    def locked(self) -> bool:
        return self._owner is not None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    # -- Condition integration ----------------------------------------- #

    def _is_owned_by(self, rec) -> bool:
        return self._owner is rec

    def _cond_release(self, sched):
        count = self._count
        self._owner = None
        self._count = 0
        sched._wake_all(self._ws)
        return count

    def _cond_reacquire(self, state) -> None:
        self.acquire()
        self._count = state


class Condition:
    """Modelled ``threading.Condition`` (wraps a modelled Lock/RLock)."""

    def __init__(self, lock=None):
        self._lock = lock if lock is not None else RLock()
        self._ws = WaitSet("Condition.wait")

    def acquire(self, *args, **kwargs):
        return self._lock.acquire(*args, **kwargs)

    def release(self):
        self._lock.release()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()
        return False

    def wait(self, timeout=None) -> bool:
        sched, rec = _ctx("Condition.wait")
        if not self._lock._is_owned_by(rec):
            raise RuntimeError("cannot wait on un-acquired lock")
        tmo = None if timeout is None else max(float(timeout), 0.0)
        if tmo == 0.0:
            # CPython models this as release + one non-blocking waiter
            # check + reacquire: an immediate poll, never a park.
            with sched._cv:
                state = self._lock._cond_release(sched)
            sched.yield_point("cond.wait")
            self._lock._cond_reacquire(state)
            return False
        with sched._cv:
            state = self._lock._cond_release(sched)
            timed_out = sched._handoff(rec, self._ws, tmo, "cond.wait")
        self._lock._cond_reacquire(state)
        return not timed_out

    def wait_for(self, predicate: Callable[[], Any], timeout=None):
        result = predicate()
        while not result:
            if not self.wait(timeout):
                return predicate()
            result = predicate()
        return result

    def notify(self, n: int = 1) -> None:
        sched, rec = _ctx("Condition.notify")
        if not self._lock._is_owned_by(rec):
            raise RuntimeError("cannot notify on un-acquired lock")
        with sched._cv:
            sched._wake_n(self._ws, n)
        sched.yield_point("cond.notify")

    def notify_all(self) -> None:
        sched, rec = _ctx("Condition.notify_all")
        if not self._lock._is_owned_by(rec):
            raise RuntimeError("cannot notify on un-acquired lock")
        with sched._cv:
            sched._wake_n(self._ws, len(self._ws.waiters))
        sched.yield_point("cond.notify")

    notifyAll = notify_all  # deprecated alias, kept for compatibility


class Event:
    """Modelled ``threading.Event``."""

    def __init__(self):
        self._flag = False
        self._ws = WaitSet("Event.wait")

    def is_set(self) -> bool:
        return self._flag

    isSet = is_set  # deprecated alias, kept for compatibility

    def set(self) -> None:
        sched = current_scheduler()
        if sched is None:
            self._flag = True  # outside any run: nobody can be parked on it
            return
        sched, rec = _ctx("Event.set")  # unmanaged thread during a run: fail
        with sched._cv:
            self._flag = True
            sched._wake_all(self._ws)
        sched.yield_point("event.set")

    def clear(self) -> None:
        sched = current_scheduler()
        if sched is None:
            self._flag = False
            return
        sched, rec = _ctx("Event.clear")
        with sched._cv:
            self._flag = False
        sched.yield_point("event.clear")

    def wait(self, timeout=None) -> bool:
        sched, rec = _ctx("Event.wait")
        sched.yield_point("event.wait")
        tmo = None if timeout is None else max(float(timeout), 0.0)
        with sched._cv:
            while not self._flag:
                if tmo == 0.0:
                    return self._flag
                if sched._handoff(rec, self._ws, tmo, "event.blocked"):
                    return self._flag  # timeout: report current flag state
        return True


class Semaphore:
    """Modelled ``threading.Semaphore``."""

    def __init__(self, value: int = 1):
        if value < 0:
            raise ValueError("semaphore initial value must be >= 0")
        self._value = value
        self._ws = WaitSet(self._describe)

    def _describe(self) -> str:
        return "Semaphore(value=%d)" % self._value

    def acquire(self, blocking: bool = True, timeout=None) -> bool:
        sched, rec = _ctx("Semaphore.acquire")
        if not blocking and timeout is not None:
            raise ValueError("can't specify timeout for non-blocking acquire")
        sched.yield_point("sem.acquire")
        tmo = None if timeout is None else max(float(timeout), 0.0)
        with sched._cv:
            while self._value == 0:
                if not blocking or tmo == 0.0:
                    return False
                if sched._handoff(rec, self._ws, tmo, "sem.blocked"):
                    return False
            self._value -= 1
        return True

    def release(self, n: int = 1) -> None:
        if n < 1:
            raise ValueError("n must be one or more")
        sched, rec = _ctx("Semaphore.release")
        with sched._cv:
            self._value += n
            sched._wake_all(self._ws)
        sched.yield_point("sem.release")

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


class BoundedSemaphore(Semaphore):
    """Modelled ``threading.BoundedSemaphore``."""

    def __init__(self, value: int = 1):
        super().__init__(value)
        self._initial = value

    def release(self, n: int = 1) -> None:
        if self._value + n > self._initial:
            raise ValueError("Semaphore released too many times")
        super().release(n)


class Thread:
    """Drop-in for ``threading.Thread(target=...)`` inside a controlled run.

    Subclassing with an overridden ``run()`` works when the subclass is
    defined while the patch is active (i.e. inside the model); code that
    defines Thread subclasses at import time keeps the real base class
    and is not controlled — prefer ``target=`` or
    :func:`interleave_test.spawn`.
    """

    def __init__(
        self,
        group=None,
        target: Optional[Callable] = None,
        name: Optional[str] = None,
        args=(),
        kwargs=None,
        *,
        daemon: Optional[bool] = None,
    ):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.name = name
        # The model's root thread is non-daemonic, so inherited daemon
        # status is deterministically False.
        self.daemon = False if daemon is None else daemon
        self._task = None

    def start(self) -> None:
        sched = current_scheduler()
        if sched is None:
            raise RuntimeError(
                "interleave_test.Thread.start() outside a controlled run"
            )
        if self._task is not None:
            raise RuntimeError("threads can only be started once")
        display = self.name or getattr(self._target, "__name__", None) or "thread"
        self._task = sched.spawn(self.run, name=display)
        if self.name is None:
            self.name = display

    def run(self) -> None:
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def join(self, timeout=None) -> None:
        if self._task is None:
            raise RuntimeError("cannot join thread before it is started")
        self._task.join(timeout)

    def is_alive(self) -> bool:
        return self._task is not None and self._task.is_alive()

    @property
    def ident(self) -> Optional[int]:
        # Deterministic surrogate: the scheduler's logical thread id.
        return self._task._rec.tid if self._task is not None else None

    @property
    def native_id(self) -> Optional[int]:
        return self.ident

    # Deprecated accessor aliases, kept for compatibility.
    def getName(self) -> Optional[str]:
        return self.name

    def setName(self, name) -> None:
        self.name = name

    def isDaemon(self) -> bool:
        return self.daemon

    def setDaemon(self, daemonic) -> None:
        self.daemon = daemonic


def sleep(secs) -> None:
    """Patched ``time.sleep``: a pure scheduling point, returns instantly.

    Inside a model, real sleeping is pointless (only one thread runs at a
    time and the schedule is explicit); a sleep becomes an extra
    opportunity for the strategy to switch threads.  It does not advance
    the virtual clock — only logical timeout expiry does.  Outside a run
    this falls through to the real ``time.sleep``.
    """
    sched = current_scheduler()
    rec = sched.me() if sched is not None else None
    if rec is None:
        _real_sleep(secs)
        return
    sched.yield_point("sleep")


def monotonic() -> float:
    """Patched ``time.monotonic``: the scheduler's virtual clock.

    Starts at 0.0 each schedule and advances only when a logical timeout
    fires, so timed waits (e.g. ``queue.Queue.get(timeout=...)``) expire
    deterministically instead of spinning against the real clock.
    Unmanaged threads (and code outside a run) get the real clock.
    """
    sched = current_scheduler()
    rec = sched.me() if sched is not None else None
    if rec is None:
        return _real_monotonic()
    return sched._vnow


def time() -> float:
    """Patched ``time.time``: fixed epoch + the virtual clock (see monotonic)."""
    sched = current_scheduler()
    rec = sched.me() if sched is not None else None
    if rec is None:
        return _real_time()
    return _TIME_EPOCH + sched._vnow
