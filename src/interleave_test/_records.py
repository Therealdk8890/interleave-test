"""Value objects: scheduling steps, recorded schedules, failures, results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Step:
    """One scheduling decision, kept for diagnostics.

    ``chosen`` is the logical thread id that was granted the next slice;
    ``location`` is where that thread resumed (best effort, ``file:line``);
    ``kind`` is the kind of scheduling point that triggered the decision
    (``"step"`` for a traced bytecode/line, otherwise the synchronization
    operation, e.g. ``"lock.release"``).
    """

    index: int
    chosen: int
    runnable: Tuple[int, ...]
    kind: str
    location: str
    thread_name: str


@dataclass
class Schedule:
    """A complete, replayable record of the scheduling decisions of one run.

    ``choices[i]`` is the logical thread id granted at decision ``i``.
    Schedules serialize to JSON so they can be stored next to a bug report
    and replayed later with :func:`interleave_test.replay`.  They are
    interpreter-version-specific (decisions count bytecode instructions),
    so the recording interpreter is stored in ``python``.
    """

    choices: List[int]
    granularity: str = "opcode"
    python: str = ""  # interpreter version the schedule was recorded on

    def __len__(self) -> int:
        return len(self.choices)

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": 1,
                "granularity": self.granularity,
                "python": self.python,
                "choices": self.choices,
            }
        )

    @classmethod
    def from_json(cls, text: str) -> "Schedule":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("schedule JSON must be an object")
        version = data.get("version", 1)
        if version != 1:
            raise ValueError(
                "schedule format version %r is not supported by this version "
                "of interleave-test (expected 1)" % (version,)
            )
        choices = data.get("choices")
        if not isinstance(choices, list) or not all(
            isinstance(c, int) and not isinstance(c, bool) for c in choices
        ):
            raise ValueError("schedule JSON must contain 'choices': a list of ints")
        granularity = data.get("granularity", "opcode")
        if granularity not in ("opcode", "line"):
            raise ValueError(
                "unknown granularity %r in schedule (expected 'opcode' or "
                "'line')" % (granularity,)
            )
        return cls(
            choices=list(choices),
            granularity=granularity,
            python=str(data.get("python", "")),
        )


@dataclass
class Failure:
    """Everything known about one failing schedule."""

    kind: str  # "exception" | "deadlock" | "hang"
    iteration: int  # 0-based index of the failing schedule within the run
    seed: Optional[int]
    schedule: Schedule
    steps: List[Step] = field(default_factory=list)
    exception: Optional[BaseException] = None
    traceback: Optional[str] = None
    details: str = ""
    leaked_threads: int = 0

    def format(self, max_steps: int = 25) -> str:
        lines = []
        if self.kind == "exception":
            name = type(self.exception).__name__ if self.exception else "exception"
            lines.append("model raised %s under an explored interleaving" % name)
        elif self.kind == "deadlock":
            lines.append("deadlock: every live thread is blocked")
        else:
            lines.append("hang: the schedule did not complete in time")
        if self.seed is None:
            lines.append(
                "replayed schedule, %d scheduling decisions" % len(self.schedule)
            )
        else:
            lines.append(
                "found on schedule %d of this run (seed=%s), %d scheduling "
                "decisions" % (self.iteration + 1, self.seed, len(self.schedule))
            )
        if self.details:
            lines.append("")
            lines.append(self.details)
        if self.traceback:
            lines.append("")
            lines.append(self.traceback.rstrip())
        if self.steps:
            shown = list(self.steps)[-max_steps:]
            lines.append("")
            lines.append(
                "last %d of %d scheduling decisions "
                "(thread granted, where it resumed):" % (len(shown), len(self.schedule))
            )
            for s in shown:
                lines.append(
                    "  #%-5d T%d(%s)  %s" % (s.index, s.chosen, s.thread_name, s.location)
                )
        if self.leaked_threads:
            lines.append("")
            lines.append(
                "warning: %d thread(s) could not be terminated and are still "
                "running in this process" % self.leaked_threads
            )
        lines.append("")
        lines.append(
            "reproduce exactly:  interleave_test.replay(model, err.failure.schedule)"
        )
        if self.seed is not None:
            lines.append(
                "rerun the search:   interleave_test.explore(model, seed=%s, ...)"
                % self.seed
            )
        return "\n".join(lines)


@dataclass
class Result:
    """Outcome of an :func:`interleave_test.explore` run."""

    schedules_run: int
    seed: Optional[int]
    failure: Optional[Failure] = None
    exhausted: bool = False  # DFS fully explored its (bounded) schedule space
    elapsed: float = 0.0
    leaked_threads: int = 0  # threads that survived teardown, whole run

    @property
    def ok(self) -> bool:
        return self.failure is None
