"""interleave-test: deterministic thread-interleaving testing for Python.

Runs a concurrent model function under a controlled scheduler that owns
every thread switch, then *searches* the space of interleavings —
randomly, with PCT priorities, or exhaustively with a preemption bound —
instead of running threads once and hoping.  Failures come back as an
exact, replayable schedule; deadlocks are detected and explained.

Quickstart::

    import interleave_test as it

    def model():
        counter = {"n": 0}

        def increment():
            counter["n"] += 1   # read-modify-write: not atomic

        t1 = it.spawn(increment)
        t2 = it.spawn(increment)
        t1.join(); t2.join()
        assert counter["n"] == 2

    it.explore(model, iterations=500)   # raises RaceFound with a schedule
"""

from ._errors import (
    DeadlockError,
    HangError,
    InterleaveError,
    RaceFound,
    ReplayDivergence,
)
from ._primitives import (
    BoundedSemaphore,
    Condition,
    Event,
    Lock,
    RLock,
    Semaphore,
    Thread,
)
from ._records import Failure, Result, Schedule, Step
from ._runner import explore, interleave, replay, spawn
from ._scheduler import Task, no_interleave
from ._strategies import (
    DFSStrategy,
    PCTStrategy,
    RandomStrategy,
    ReplayStrategy,
    Strategy,
)

__version__ = "0.1.0"

__all__ = [
    "explore",
    "replay",
    "spawn",
    "interleave",
    "no_interleave",
    "Task",
    "Lock",
    "RLock",
    "Condition",
    "Event",
    "Semaphore",
    "BoundedSemaphore",
    "Thread",
    "Strategy",
    "RandomStrategy",
    "PCTStrategy",
    "DFSStrategy",
    "ReplayStrategy",
    "Schedule",
    "Step",
    "Failure",
    "Result",
    "InterleaveError",
    "RaceFound",
    "DeadlockError",
    "HangError",
    "ReplayDivergence",
    "__version__",
]
