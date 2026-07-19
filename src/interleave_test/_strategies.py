"""Scheduling strategies: how the next thread is chosen at each decision.

A strategy sees every scheduling point at which more than one thread
could run and picks the one that runs next.  Strategies are consulted
only at genuine decision points; forced choices (a single runnable
thread) are recorded but not delegated.
"""

from __future__ import annotations

import random
from typing import List, Optional

from ._errors import ReplayDivergence


class Strategy:
    """Base class for scheduling strategies.

    ``pick`` receives the runnable thread records (each has ``.tid`` and
    ``.name``), the record of the thread that just yielded if it is still
    runnable (``current``, or None if it blocked or finished), and the
    global decision index.

    A strategy may also define an optional ``observe(chosen, index)``
    method; the scheduler calls it for *forced* decisions (exactly one
    runnable thread), which never reach ``pick``.  ReplayStrategy uses it
    to validate forced decisions against the recording.
    """

    exhausted = False

    def more(self) -> Optional[bool]:
        """Return False when no further schedules remain (DFS);
        None when the strategy is unbounded and ``iterations`` governs."""
        return None

    def begin(self, iteration: int, seed: int) -> None:
        """Called before each schedule with a per-schedule derived seed."""

    def pick(self, runnable, current, index: int):
        raise NotImplementedError

    def end(self) -> None:
        """Called after each schedule completes (clean or failing)."""


class RandomStrategy(Strategy):
    """Uniform random choice among runnable threads at every decision."""

    def begin(self, iteration: int, seed: int) -> None:
        self._rng = random.Random(seed)

    def pick(self, runnable, current, index: int):
        return self._rng.choice(runnable)


class PCTStrategy(Strategy):
    """Probabilistic Concurrency Testing (Burckhardt et al., ASPLOS 2010).

    Each schedule assigns every thread a random base priority and always
    runs the highest-priority runnable thread; at ``depth - 1`` randomly
    chosen decision indices the running thread is demoted below all
    others.  For a bug that needs d ordering constraints among n threads
    and k steps, one schedule finds it with probability >= 1/(n * k^(d-1))
    — much better than uniform random for rare orderings.
    """

    def __init__(self, depth: int = 3):
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self._max_steps = 64

    def begin(self, iteration: int, seed: int) -> None:
        rng = self._rng = random.Random(seed)
        self._prio = {}
        self._demote_points = sorted(
            rng.randrange(self._max_steps) for _ in range(self.depth - 1)
        )
        self._next_demotion = -1.0
        self._seen = 0

    def pick(self, runnable, current, index: int):
        self._seen = max(self._seen, index + 1)
        # A demote point that landed on a forced decision (which never
        # reaches pick) fires at the next real decision instead of being
        # silently lost.
        if (
            self._demote_points
            and index >= self._demote_points[0]
            and current is not None
        ):
            self._demote_points.pop(0)
            self._prio[current.tid] = self._next_demotion
            self._next_demotion -= 1.0
        for r in runnable:
            if r.tid not in self._prio:
                self._prio[r.tid] = self._rng.random() + 1.0
        return max(runnable, key=lambda r: self._prio[r.tid])

    def end(self) -> None:
        # Adapt the demotion-point range to the schedules actually seen.
        if self._seen > self._max_steps:
            self._max_steps = self._seen


class DFSStrategy(Strategy):
    """Bounded exhaustive search over schedules (CHESS-style).

    Uses a continue-current default (run the yielding thread on, i.e. no
    preemption) and a DFS frontier of untried branch points.  Only
    schedules with at most ``max_preemptions`` preemptions are explored:
    a preemption is switching away from a thread that could have
    continued.  Most real concurrency bugs need very few preemptions
    (CHESS found most with <= 2), which keeps the space tractable.

    Requires a deterministic model: on a revisit the recorded prefix must
    replay identically.
    """

    def __init__(self, max_preemptions: int = 2):
        self.max_preemptions = max_preemptions
        self._frontier = [([], 0)]  # (prefix of chosen tids, preemptions in prefix)
        self._trace = None

    def more(self) -> Optional[bool]:
        return bool(self._frontier)

    def begin(self, iteration: int, seed: int) -> None:
        self._prefix, _ = self._frontier.pop()
        self._trace = []  # (runnable tids, chosen tid, current tid or None)
        self._precounts = []  # preemptions before decision i
        self._pre = 0

    def pick(self, runnable, current, index: int):
        i = len(self._trace)
        cur_tid = current.tid if current is not None else None
        if i < len(self._prefix):
            want = self._prefix[i]
            chosen = next((r for r in runnable if r.tid == want), None)
            if chosen is None:
                raise ReplayDivergence(
                    "DFS revisit diverged: T%d not runnable at decision %d; the "
                    "model is nondeterministic (wall clock, unseeded randomness, "
                    "or unmodelled synchronization)" % (want, i)
                )
        elif current is not None:
            chosen = current
        else:
            chosen = min(runnable, key=lambda r: r.tid)
        self._precounts.append(self._pre)
        if cur_tid is not None and chosen.tid != cur_tid:
            self._pre += 1
        self._trace.append((tuple(r.tid for r in runnable), chosen.tid, cur_tid))
        return chosen

    def end(self) -> None:
        trace = self._trace or []
        chosen_seq = [c for (_, c, _) in trace]
        # Each execution owns the decisions beyond its prefix: enqueue every
        # untried alternative there that stays within the preemption bound.
        for i in range(len(trace) - 1, len(self._prefix) - 1, -1):
            runnable, chosen, cur = trace[i]
            base = self._precounts[i]
            for alt in runnable:
                if alt == chosen:
                    continue
                extra = 1 if (cur is not None and alt != cur) else 0
                if base + extra <= self.max_preemptions:
                    self._frontier.append((chosen_seq[:i] + [alt], base + extra))
        if not self._frontier:
            self.exhausted = True


class ReplayStrategy(Strategy):
    """Follow a previously recorded :class:`~interleave_test.Schedule` exactly."""

    def __init__(self, schedule):
        self._choices = list(schedule.choices)

    def pick(self, runnable, current, index: int):
        if index < len(self._choices):
            want = self._choices[index]
            for r in runnable:
                if r.tid == want:
                    return r
            raise ReplayDivergence(
                "replay diverged at decision %d: T%d is not runnable (the "
                "model must be deterministic apart from scheduling, and the "
                "schedule must have been recorded on this Python version)"
                % (index, want)
            )
        # Past the recorded schedule (a complete recording should not get
        # here): fall back to the continue-current default.
        if current is not None:
            return current
        return min(runnable, key=lambda r: r.tid)

    def observe(self, chosen, index: int) -> None:
        # Forced decisions never reach pick(); validating them too turns a
        # silently-wrong replay into a diagnosed divergence.
        if index < len(self._choices) and self._choices[index] != chosen.tid:
            raise ReplayDivergence(
                "replay diverged at decision %d: the recording chose T%d but "
                "T%d was the only runnable thread (model nondeterminism, or a "
                "different Python version than the schedule was recorded on)"
                % (index, self._choices[index], chosen.tid)
            )
