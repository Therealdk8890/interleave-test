# Changelog

## 0.1.0 (2026-07-19)

Initial release.

- Cooperative scheduler serializing controlled threads with direct
  thread-to-thread handoff; scheduling points at every bytecode
  instruction (or line) of user code via `sys.monitoring` (PEP 669),
  plus every modelled synchronization operation.
- Strategies: `random`, `pct` (probabilistic concurrency testing),
  `dfs` (bounded-exhaustive, CHESS-style preemption bound), custom
  `Strategy` subclasses.
- Deterministic replay: failures carry a JSON-serializable `Schedule`;
  `replay()` re-runs it decision for decision.
- Deadlock detection with a blocked-threads table; per-schedule hang
  watchdog.
- Modelled primitives: `Lock`, `RLock`, `Condition`, `Event`,
  `Semaphore`, `BoundedSemaphore`, `Thread`, `time.sleep`; the
  `threading` names are patched inside a run, which makes
  `queue.Queue` work unmodified.
- `spawn()`, `no_interleave()`, `@interleave` decorator, `explore()` /
  `replay()` entry points.
- Deterministic virtual clock: patched `time.monotonic`/`time.time`
  advance when logical timeouts fire, so `queue.Queue.get(timeout=...)`
  and friends expire deterministically instead of spinning.
- Works identically on GIL and free-threaded (3.13t/3.14t) builds
  (verified: byte-for-byte identical failure output on 3.14 and 3.14t).
