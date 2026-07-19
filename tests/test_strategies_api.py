"""Strategy behaviour and API-surface checks."""

import pytest

import interleave_test as it
from interleave_test import RaceFound

from test_races import unsafe_counter_model


def test_pct_strategy_finds_race():
    with pytest.raises(RaceFound):
        it.explore(unsafe_counter_model, iterations=300, seed=42, strategy="pct")


def test_dfs_finds_race_and_is_deterministic():
    def once():
        try:
            it.explore(
                unsafe_counter_model,
                iterations=5000,
                strategy="dfs",
                max_preemptions=1,
            )
        except RaceFound as e:
            return (e.failure.iteration, tuple(e.failure.schedule.choices))
        return None

    first = once()
    assert first is not None
    assert first == once()  # DFS ignores the seed: fully deterministic


def test_dfs_exhausts_small_model():
    def model():
        items = []

        def a():
            items.append(1)

        def b():
            items.append(2)

        t1 = it.spawn(a)
        t2 = it.spawn(b)
        t1.join()
        t2.join()
        assert len(items) == 2

    res = it.explore(model, iterations=100000, strategy="dfs", max_preemptions=1)
    assert res.ok
    assert res.exhausted
    assert res.schedules_run > 1


def test_interleave_decorator():
    @it.interleave(iterations=200, seed=1234)
    def decorated():
        unsafe_counter_model()

    with pytest.raises(RaceFound):
        decorated()
    assert callable(decorated.model)


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        it.explore(unsafe_counter_model, strategy="quantum")


def test_bad_granularity_rejected():
    with pytest.raises(ValueError):
        it.explore(unsafe_counter_model, granularity="atom")


def test_primitive_outside_run_raises():
    lock = it.Lock()
    with pytest.raises(RuntimeError):
        lock.acquire()


def test_spawn_outside_run_raises():
    with pytest.raises(RuntimeError):
        it.spawn(lambda: None)


def test_nested_explore_rejected():
    def model():
        it.explore(lambda: None, iterations=1)

    with pytest.raises(RaceFound) as ei:
        it.explore(model, iterations=1, seed=0)
    assert isinstance(ei.value.__cause__, RuntimeError)


def test_failure_format_mentions_seed_and_replay():
    with pytest.raises(RaceFound) as ei:
        it.explore(unsafe_counter_model, iterations=200, seed=77)
    text = str(ei.value)
    assert "seed=77" in text
    assert "replay" in text
