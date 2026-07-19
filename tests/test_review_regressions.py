"""Regression tests for defects found in the adversarial review."""

import queue
import threading
import time

import pytest

import interleave_test as it
from interleave_test import DeadlockError, HangError, RaceFound, ReplayDivergence

from test_races import unsafe_counter_model


class NoisyError(Exception):
    """An exception whose __str__ is traced user code."""

    def __str__(self):
        parts = []
        for piece in ("boom", "with", "traced", "__str__"):
            parts.append(piece)
        return " ".join(parts)


def test_traced_dunder_str_does_not_deadlock_failure_reporting():
    # _fail() used to format the traceback (which calls str(exc)) while
    # holding the scheduler cv; a traced __str__ then self-deadlocked.
    def model():
        def worker():
            raise NoisyError()

        t = it.spawn(worker)
        t.join()

    started = time.monotonic()
    with pytest.raises(RaceFound) as ei:
        it.explore(model, iterations=5, seed=0, timeout=10.0)
    assert time.monotonic() - started < 10
    assert isinstance(ei.value.__cause__, NoisyError)
    assert "traced __str__" in ei.value.failure.traceback


def test_deadlock_teardown_is_fast_and_threads_die():
    # Aborted-while-blocked threads used to be resurrected by teardown
    # wake-ups, burning the full 5s grace period per deadlock.
    tasks = []

    def model():
        a = it.Lock()
        b = it.Lock()

        def one():
            with a:
                with b:
                    pass

        def two():
            with b:
                with a:
                    pass

        tasks.clear()
        tasks.append(it.spawn(one))
        tasks.append(it.spawn(two))
        for t in tasks:
            t.join()

    started = time.monotonic()
    with pytest.raises(DeadlockError):
        it.explore(model, iterations=5000, strategy="dfs", max_preemptions=1)
    assert time.monotonic() - started < 2.0
    assert all(not t.is_alive() for t in tasks)


def test_no_interleave_region_is_still_abortable():
    # The abort latch used to be checked after no_yield, making a thread
    # spinning inside no_interleave() impossible to tear down.
    def model():
        def spinner():
            with it.no_interleave():
                while True:
                    pass

        t = it.spawn(spinner)
        t.join()

    started = time.monotonic()
    with pytest.raises(HangError):
        it.explore(model, iterations=1, seed=0, timeout=2.0)
    assert time.monotonic() - started < 10
    # the process is still usable afterwards
    assert it.explore(lambda: None, iterations=2, seed=0).ok


def test_simplequeue_is_modelled():
    # queue.SimpleQueue is C-implemented; unpatched, its get() blocks the
    # OS thread invisibly and the model dies as a hang.
    def model():
        sq = queue.SimpleQueue()
        got = []

        def producer():
            sq.put("x")

        def consumer():
            got.append(sq.get())

        c = it.spawn(consumer)
        p = it.spawn(producer)
        p.join()
        c.join()
        assert got == ["x"]

    assert it.explore(model, iterations=30, seed=4).ok


def test_queue_get_timeout_uses_virtual_clock():
    # queue.Queue.get(timeout=...) computes a deadline from time.monotonic;
    # with a real clock it spins against logical timeouts.
    def model():
        q = queue.Queue()
        with pytest.raises(queue.Empty):
            q.get(timeout=5.0)  # virtual seconds: expires immediately

    started = time.monotonic()
    assert it.explore(model, iterations=20, seed=0).ok
    assert time.monotonic() - started < 5


def test_compat_aliases_exist():
    def model():
        ev = it.Event()
        cond = it.Condition()
        rlock = it.RLock()
        assert ev.isSet() is False
        with cond:
            cond.notifyAll()
        assert rlock.locked() is False
        with rlock:
            assert rlock.locked()
        t = threading.Thread(target=lambda: None, name="worker")
        assert t.getName() == "worker"
        t.setDaemon(True)
        assert t.isDaemon()
        assert t.ident is None
        t.start()
        t.join()
        assert isinstance(t.ident, int)

    assert it.explore(model, iterations=5, seed=0).ok


def test_primitive_validation_matches_threading():
    def model():
        lock = it.Lock()
        with pytest.raises(ValueError):
            lock.acquire(blocking=False, timeout=1.0)
        with pytest.raises(ValueError):
            lock.acquire(timeout=-2)
        sem = it.Semaphore(1)
        with pytest.raises(ValueError):
            sem.acquire(blocking=False, timeout=1.0)
        with pytest.raises(ValueError):
            sem.release(0)

    assert it.explore(model, iterations=3, seed=0).ok


def test_zero_timeout_is_a_poll_not_a_park():
    def model():
        lock = it.Lock()
        held = it.Event()
        release = it.Event()

        def holder():
            with lock:
                held.set()
                release.wait()

        t = it.spawn(holder)
        held.wait()
        assert lock.acquire(timeout=0) is False  # immediate, no deadlock
        assert it.Semaphore(0).acquire(timeout=0) is False
        ev = it.Event()
        assert ev.wait(timeout=0) is False
        cond = it.Condition()
        with cond:
            assert cond.wait(timeout=0) is False
        release.set()
        t.join()

    assert it.explore(model, iterations=30, seed=6).ok


def test_interleave_decorator_returns_none_and_rejects_params():
    @it.interleave(iterations=3, seed=0)
    def decorated():
        pass

    assert decorated() is None
    assert decorated.last_result is not None
    assert decorated.last_result.ok

    with pytest.raises(TypeError, match="no required parameters"):

        @it.interleave(iterations=1)
        def with_fixture(tmp_path):
            pass


def test_schedule_from_json_validation():
    with pytest.raises(ValueError, match="version"):
        it.Schedule.from_json('{"version": 2, "choices": [0]}')
    with pytest.raises(ValueError, match="choices"):
        it.Schedule.from_json('{"version": 1}')
    with pytest.raises(ValueError, match="choices"):
        it.Schedule.from_json('{"version": 1, "choices": "abc"}')
    with pytest.raises(ValueError, match="granularity"):
        it.Schedule.from_json('{"version": 1, "choices": [0], "granularity": "x"}')
    with pytest.raises(ValueError, match="object"):
        it.Schedule.from_json("[1, 2]")
    # missing version is treated as 1 for compatibility
    sched = it.Schedule.from_json('{"choices": [0, 1]}')
    assert sched.choices == [0, 1]


def test_replay_failure_reports_no_fake_seed():
    with pytest.raises(RaceFound) as ei:
        it.explore(unsafe_counter_model, iterations=200, seed=7)
    with pytest.raises(RaceFound) as ei2:
        it.replay(unsafe_counter_model, ei.value.failure.schedule)
    text = str(ei2.value)
    assert "replayed schedule" in text
    assert "rerun the search" not in text
    # replaying via the Failure keeps the original seed for the rerun hint
    with pytest.raises(RaceFound) as ei3:
        it.replay(unsafe_counter_model, ei.value.failure)
    assert "seed=7" in str(ei3.value)


def test_corrupted_schedule_diverges_not_silently():
    with pytest.raises(RaceFound) as ei:
        it.explore(unsafe_counter_model, iterations=200, seed=7)
    sched = ei.value.failure.schedule
    corrupted = it.Schedule(
        choices=[99] + list(sched.choices[1:]),
        granularity=sched.granularity,
        python=sched.python,
    )
    with pytest.raises(ReplayDivergence):
        it.replay(unsafe_counter_model, corrupted)


def test_include_accepts_bare_string():
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    with pytest.raises(RaceFound):
        it.explore(unsafe_counter_model, iterations=200, seed=1234, include=here)


def test_exclude_wins_inside_include():
    # With this test file excluded, the counter increment is untraced and
    # effectively atomic: the race disappears.
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    res = it.explore(
        unsafe_counter_model,
        iterations=200,
        seed=1234,
        include=here,
        exclude=here,
    )
    assert res.ok


def test_exhausted_dfs_instance_is_rejected_on_reuse():
    strat = it.DFSStrategy(max_preemptions=0)

    def model():
        t = it.spawn(lambda: None)
        t.join()

    res = it.explore(model, iterations=100000, strategy=strat)
    assert res.exhausted
    with pytest.raises(RuntimeError, match="no schedules left"):
        it.explore(model, iterations=10, strategy=strat)


def test_unmanaged_thread_touching_primitive_fails_fast():
    # A modelled primitive touched from a thread the scheduler doesn't
    # manage must fail the schedule immediately, not hang for 30s.
    import _thread

    def model():
        ev = it.Event()

        def rogue():  # raw OS thread: invisible to the scheduler
            try:
                ev.set()
            except RuntimeError:
                pass  # also raised at the call site, expected

        _thread.start_new_thread(rogue, ())
        while True:  # stay runnable until the external error aborts us
            pass

    started = time.monotonic()
    with pytest.raises(RaceFound) as ei:
        it.explore(model, iterations=3, seed=0, timeout=15.0)
    assert time.monotonic() - started < 10
    assert "not managed by interleave_test" in str(ei.value.__cause__)