"""The classic lost-update race, found deterministically.

Run:  python examples/counter_race.py
"""

import interleave_test as it


class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1  # LOAD value / ADD 1 / STORE value: three steps, no lock


def model():
    counter = Counter()
    t1 = it.spawn(counter.increment)
    t2 = it.spawn(counter.increment)
    t1.join()
    t2.join()
    assert counter.value == 2, "lost update: both threads read the same old value"


if __name__ == "__main__":
    it.explore(model, iterations=500, seed=42)
