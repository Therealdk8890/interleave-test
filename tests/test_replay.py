"""Determinism: seeds reproduce searches, schedules replay exactly."""

import pytest

import interleave_test as it
from interleave_test import RaceFound

from test_races import unsafe_counter_model


def test_replay_reproduces_failure():
    with pytest.raises(RaceFound) as ei:
        it.explore(unsafe_counter_model, iterations=200, seed=7)
    schedule = ei.value.failure.schedule

    with pytest.raises(RaceFound) as ei2:
        it.replay(unsafe_counter_model, schedule)
    assert isinstance(ei2.value.failure.exception, AssertionError)


def test_replay_accepts_failure_and_json():
    with pytest.raises(RaceFound) as ei:
        it.explore(unsafe_counter_model, iterations=200, seed=7)
    failure = ei.value.failure

    with pytest.raises(RaceFound):
        it.replay(unsafe_counter_model, failure)  # a Failure

    text = failure.schedule.to_json()
    assert isinstance(text, str)
    with pytest.raises(RaceFound):
        it.replay(unsafe_counter_model, text)  # a JSON string


def test_same_seed_same_result():
    def run():
        try:
            it.explore(unsafe_counter_model, iterations=200, seed=99)
        except RaceFound as e:
            return (e.failure.iteration, tuple(e.failure.schedule.choices))
        return None

    first = run()
    assert first is not None
    assert first == run()


def test_schedule_json_roundtrip():
    sched = it.Schedule(choices=[0, 1, 1, 0, 2], granularity="line")
    back = it.Schedule.from_json(sched.to_json())
    assert back.choices == sched.choices
    assert back.granularity == "line"
