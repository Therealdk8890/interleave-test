"""Exception types raised by interleave-test."""

from __future__ import annotations


class InterleaveError(Exception):
    """Base class for all interleave-test errors.

    Errors raised by :func:`interleave_test.explore` carry two extra
    attributes:

    * ``failure`` — the :class:`interleave_test.Failure` describing what
      went wrong, including the recorded :class:`~interleave_test.Schedule`.
    * ``result`` — the :class:`interleave_test.Result` for the whole run.
    """

    failure = None
    result = None


class RaceFound(InterleaveError):
    """An interleaving was found under which the model raised an exception.

    The original exception is available as ``__cause__`` and as
    ``err.failure.exception``.
    """


class DeadlockError(InterleaveError):
    """An interleaving was found under which every live thread was blocked."""


class HangError(InterleaveError):
    """A single schedule exceeded the per-schedule wall-clock timeout.

    This usually means the model performed a real blocking operation the
    scheduler does not model (C-level I/O, an unpatched primitive, ...).
    """


class ReplayDivergence(InterleaveError):
    """A recorded schedule no longer matches the model's behaviour.

    Replay requires the model to be deterministic apart from thread
    scheduling: no wall-clock reads, no unseeded randomness, no
    unmodelled synchronization.  Schedules are also interpreter-version-
    specific (decisions count bytecode instructions), so replay on the
    Python version that recorded them.
    """
