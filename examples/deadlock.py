"""An ABBA deadlock, found and explained by bounded-exhaustive search.

Run:  python examples/deadlock.py
"""

import interleave_test as it


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


if __name__ == "__main__":
    # One preemption is enough: run t1 past `with a:`, switch to t2.
    it.explore(model, strategy="dfs", max_preemptions=1, iterations=10_000)
