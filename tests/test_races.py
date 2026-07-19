"""End-to-end: the explorer finds real races and proves fixes."""

import pytest

import interleave_test as it
from interleave_test import DeadlockError, RaceFound


class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1  # read-modify-write: not atomic across threads


def unsafe_counter_model():
    c = Counter()
    t1 = it.spawn(c.increment)
    t2 = it.spawn(c.increment)
    t1.join()
    t2.join()
    assert c.value == 2


def test_finds_lost_update_race():
    with pytest.raises(RaceFound) as ei:
        it.explore(unsafe_counter_model, iterations=200, seed=1234)
    f = ei.value.failure
    assert f.kind == "exception"
    assert isinstance(f.exception, AssertionError)
    assert len(f.schedule.choices) > 0
    assert isinstance(ei.value.__cause__, AssertionError)


def test_lock_fixes_race():
    def model():
        c = Counter()
        lock = it.Lock()

        def inc():
            with lock:
                c.increment()

        t1 = it.spawn(inc)
        t2 = it.spawn(inc)
        t1.join()
        t2.join()
        assert c.value == 2

    res = it.explore(model, iterations=150, seed=1234)
    assert res.ok
    assert res.schedules_run == 150


def test_no_interleave_makes_region_atomic():
    def model():
        c = Counter()

        def inc():
            with it.no_interleave():
                c.increment()

        t1 = it.spawn(inc)
        t2 = it.spawn(inc)
        t1.join()
        t2.join()
        assert c.value == 2

    assert it.explore(model, iterations=200, seed=1234).ok


def test_check_then_act_double_init():
    def model():
        state = {"instance": None, "inits": 0}

        def get_instance():
            if state["instance"] is None:  # check ...
                state["inits"] += 1  # ... then act: two threads can get here
                state["instance"] = object()

        t1 = it.spawn(get_instance)
        t2 = it.spawn(get_instance)
        t1.join()
        t2.join()
        assert state["inits"] == 1

    with pytest.raises(RaceFound):
        it.explore(model, iterations=300, seed=21)


def test_line_granularity_finds_multi_line_race():
    def model():
        state = {"instance": None, "inits": 0}

        def get_instance():
            if state["instance"] is None:
                state["inits"] += 1
                state["instance"] = object()

        t1 = it.spawn(get_instance)
        t2 = it.spawn(get_instance)
        t1.join()
        t2.join()
        assert state["inits"] == 1

    with pytest.raises(RaceFound):
        it.explore(model, iterations=300, seed=21, granularity="line")


def test_detects_abba_deadlock():
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

        t1 = it.spawn(one)
        t2 = it.spawn(two)
        t1.join()
        t2.join()

    with pytest.raises(DeadlockError) as ei:
        it.explore(model, iterations=5000, strategy="dfs", max_preemptions=1)
    assert ei.value.failure.kind == "deadlock"
    assert "blocked on" in str(ei.value)


def test_deterministic_failure_reported_on_first_schedule():
    def model():
        raise ValueError("boom")

    with pytest.raises(RaceFound) as ei:
        it.explore(model, iterations=10, seed=0)
    assert ei.value.failure.iteration == 0
    assert isinstance(ei.value.__cause__, ValueError)


def test_raise_on_failure_false_returns_result():
    res = it.explore(
        unsafe_counter_model, iterations=200, seed=1234, raise_on_failure=False
    )
    assert not res.ok
    assert res.failure.kind == "exception"
    assert res.schedules_run >= 1
