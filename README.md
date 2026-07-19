# interleave-test

**Deterministic thread-interleaving testing for Python — find the race before free-threading finds it for you.**

Free-threaded Python (PEP 703, shipping in 3.13/3.14) removes the GIL, and
with it the accidental serialization that kept most pure-Python race
conditions asleep. The races were always there — `x += 1` has been three
bytecodes all along — but your tests ran them one thread switch per 5 ms and
never saw the bad ordering. Running tests "in parallel and hoping"
(the `pytest-run-parallel` approach) samples a handful of schedules out of
millions. Rust solved this problem with
[loom](https://github.com/tokio-rs/loom): take control of the scheduler and
*search* the interleaving space. Python has had nothing comparable.

`interleave-test` is that tool:

- **Controlled scheduler** — your model's threads are serialized and every
  bytecode instruction (or line) plus every synchronization operation is a
  switch point the *tool* chooses, not the OS.
- **Search strategies** — uniform random, PCT (probabilistic concurrency
  testing, far better at rare orderings), and bounded-exhaustive DFS in the
  style of Microsoft's CHESS (`max_preemptions`).
- **Deterministic replay** — a failure is not a flake; it's a `Schedule` you
  can serialize to JSON, attach to a bug report, and re-run instruction for
  instruction (on the same Python version — schedules count bytecode
  instructions).
- **Deadlock detection** — the scheduler always knows what every thread is
  blocked on; ABBA deadlocks come back as a table, not a hung CI job.
- **Modelled primitives** — `threading.Lock/RLock/Condition/Event/Semaphore/
  BoundedSemaphore/Thread`, `time.sleep/monotonic/time`, and
  `queue.SimpleQueue` are patched inside the run; `queue.Queue` works because
  it builds on those. Blocked threads never block an OS thread, and timed
  waits run on a deterministic virtual clock.
- **Works on GIL and free-threaded builds** — the scheduler serializes
  execution itself, so results are identical on 3.13t/3.14t and normal
  builds. A race found here is a real race under free-threading (and, at
  bytecode boundaries, under the GIL too).

Requires Python ≥ 3.12 (built on `sys.monitoring`, PEP 669). Pure Python, no
dependencies.

```
pip install interleave-test
```

(Not yet on PyPI? Install from a source checkout with `pip install .`.)

## Sixty seconds

```python
import interleave_test as it

class Counter:
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1        # LOAD / ADD / STORE — three steps, no lock

def model():
    counter = Counter()
    t1 = it.spawn(counter.increment)
    t2 = it.spawn(counter.increment)
    t1.join(); t2.join()
    assert counter.value == 2

it.explore(model, iterations=500, seed=42)
```

```
interleave_test._errors.RaceFound: model raised AssertionError under an explored interleaving
found on schedule 6 of this run (seed=42), 65 scheduling decisions

raised in T0(model)

Traceback (most recent call last):
  ...
  File "examples/counter_race.py", line 23, in model
    assert counter.value == 2
AssertionError

last 25 of 65 scheduling decisions (thread granted, where it resumed):
  #40    T2(increment)  counter_race.py:14
  #41    T2(increment)  counter_race.py:14
  #42    T0(model)      counter_race.py:21
  ...

reproduce exactly:  interleave_test.replay(model, err.failure.schedule)
rerun the search:   interleave_test.explore(model, seed=42, ...)
```

Both threads read `0`, both wrote `1`. Schedule 6, a few milliseconds in.
The same model with a lock around the increment passes every schedule —
which is now a *checkable* claim, not a hope:

```python
def model():
    counter = Counter()
    lock = it.Lock()           # or threading.Lock(): patched inside the run

    def safe_increment():
        with lock:
            counter.increment()

    t1 = it.spawn(safe_increment)
    t2 = it.spawn(safe_increment)
    t1.join(); t2.join()
    assert counter.value == 2

result = it.explore(model, iterations=500, seed=42)
assert result.ok
```

## Writing models

A **model** is a zero-argument callable that builds its own state, starts
threads, joins them, and asserts. It is run once per schedule, so it must be
deterministic apart from scheduling: no wall-clock reads, no unseeded
randomness, no I/O. (That determinism is what makes replay exact.)

Inside a model you can use:

- `it.spawn(fn, *args, name=..., **kwargs)` → `Task` with `.join()`,
  `.result()`, `.is_alive()`
- `threading.Thread(target=...)` — patched to a controlled thread
- `it.Lock/RLock/Condition/Event/Semaphore/BoundedSemaphore` — or the
  `threading.` names, which are patched inside the run
- `queue.Queue`, `LifoQueue`, `PriorityQueue` — they build on patched
  `threading` internals — and `queue.SimpleQueue`, which is patched to the
  pure-Python implementation (the C one blocks invisibly)
- `time.sleep(n)` — patched to a pure scheduling point (returns instantly;
  sleeping is meaningless when the schedule is explicit); `time.monotonic`
  and `time.time` serve the scheduler's virtual clock, so
  `q.get(timeout=5)` expires logically instead of spinning for 5 real
  seconds
- `it.no_interleave()` — a context manager that makes a region atomic, for
  setup code or for asserting "this region is the racy one"

As a pytest test, use the decorator:

```python
import interleave_test as it

@it.interleave(iterations=500, seed=0)
def test_counter_is_thread_safe():
    counter = Counter()
    lock = it.Lock()

    def safe_increment():
        with lock:
            counter.increment()

    tasks = [it.spawn(safe_increment) for _ in range(2)]
    for t in tasks:
        t.join()
    assert counter.value == 2
```

The decorated function *is* the exploration; bare `pytest` runs it (drop the
lock and pytest reports the `RaceFound`). Models take no parameters —
fixtures aren't supported; use closure state. The undecorated model stays
available as `test_counter_is_thread_safe.model`, and the last run's
`Result` as `test_counter_is_thread_safe.last_result`.

## Strategies

| strategy | what it does | when |
|---|---|---|
| `"random"` (default) | uniform choice among runnable threads at every decision | fast smoke-fuzzing |
| `"pct"` | random thread priorities + `depth-1` demotion points (Burckhardt et al.) | rare orderings; probabilistic guarantees |
| `"dfs"` | bounded-exhaustive enumeration, ≤ `max_preemptions` preemptions (CHESS-style) | small models; proof-shaped confidence |

```python
it.explore(model, strategy="dfs", max_preemptions=2, iterations=100_000)
```

DFS sets `result.exhausted = True` when it fully explored the bounded
space — for a small model that is a *theorem* about every schedule with at
most N preemptions, at the granularity you chose. `iterations` caps the
number of schedules for every strategy. Custom strategies subclass
`it.Strategy` and implement `pick(runnable, current, index)`.

## Failures, seeds, replay

Every failure raises `RaceFound`, `DeadlockError`, or `HangError`
(subclasses of `InterleaveError`) carrying:

- `err.failure.schedule` — the exact `Schedule` (JSON-serializable:
  `schedule.to_json()` / `Schedule.from_json(text)`)
- `err.failure.steps` — the last scheduling decisions, human-readable
- `err.failure.seed`, `err.result` — enough to rerun the whole search

```python
try:
    it.explore(model, iterations=1000)
except it.RaceFound as err:
    it.replay(model, err.failure.schedule)   # same crash, every time
```

Deadlocks come back explained — the failure includes a blocked-threads
table:

```
deadlock: every live thread is blocked
found on schedule 50 of this run (seed=3586340675), 79 scheduling decisions

  T0(model)  blocked on join(T1(one))  at deadlock.py:25
  T1(one)  blocked on Lock held by T2(two)  at deadlock.py:15
  T2(two)  blocked on Lock held by T1(one)  at deadlock.py:20
  ...
```

## How it works

`explore()` runs your model repeatedly. Each run, a cooperative scheduler
serializes all controlled threads: exactly one runs at a time, and at every
scheduling point — each bytecode instruction of *your* code (via
`sys.monitoring`, PEP 669), plus every modelled synchronization operation —
the strategy picks who runs next. Handoff is direct thread-to-thread; a
logically blocked thread parks inside the scheduler, never on an OS
primitive, so the runnable set is always exact. That yields deadlock
detection for free, and it makes an execution a pure function of the
decision sequence — hence replay.

`sys.monitoring` rather than `sys.settrace` is a correctness decision, not a
style one: the legacy trace hook applies per-code-object instrumentation
lazily, so the first-ever execution of a function delivers fewer events than
later ones — which silently breaks replay and DFS revisits. `sys.monitoring`
delivers identical events from the first execution on.

By default your code is traced and the stdlib/site-packages are not (their
calls execute atomically, except for the modelled primitives, which are
explicit scheduling points). Narrow or widen with `include=`/`exclude=`
path prefixes. `granularity="opcode"` (default) catches single-line races
like `x += 1`; `granularity="line"` is faster when line-atomicity is
acceptable.

## Limitations (honest ones)

- **Pure-Python interleavings only.** C-extension internals (numpy kernels,
  C-implemented containers) execute atomically; races inside them are out of
  scope. Note `x += 1` *is* pure Python — most application-level races are.
- **Models must be deterministic** apart from scheduling; replay and DFS
  depend on it. Seed your randomness, don't read clocks (`time.monotonic`/
  `time.time` are already virtualized for you).
- **Schedules are interpreter-version-specific.** Decisions count bytecode
  instructions, and bytecode changes between CPython releases; a schedule
  recorded on 3.14 will diverge on 3.12. Replay on the recording version
  (it's stored in the schedule, and `replay()` warns on mismatch).
- **`concurrent.futures.ThreadPoolExecutor` is not modelled yet** (its
  workers park on machinery created at import time, before patching). Same
  for Thread subclasses defined outside the run — prefer `target=` or
  `it.spawn`.
- **Logical timeouts.** A positive `acquire(timeout=...)`/`wait(timeout=...)`
  expires only when no thread can run, jumping the virtual clock to its
  deadline; a zero/negative timeout is an immediate poll. Timeouts order
  deterministically, not by wall clock.
- **Bounded search is bounded.** DFS explores every schedule within its
  preemption bound and granularity; it is not full DPOR, and random/PCT are
  probabilistic. Absence of a failure is evidence, not proof.
- **One exploration at a time** per process (`explore()` is not reentrant).

## Why not just run tests in parallel?

`pytest-run-parallel` and friends answer a different question: "does my
suite happen to fail when threads run concurrently on this machine today?"
The OS scheduler switches rarely and roughly uniformly; the pathological
window in your increment is three bytecodes wide. You can run a racy test
ten thousand times in parallel and never once see the lost update — then
ship, and see it weekly in production. Controlled-scheduler testing inverts
the odds: every schedule is chosen adversarially, failures are replayable,
and "no failure" can be made to mean "no failing schedule within this
bound".

## Roadmap

- Dynamic partial-order reduction (DPOR) to prune equivalent schedules
- `ThreadPoolExecutor` / `concurrent.futures` modelling
- A pytest plugin (`--interleave` to re-run marked tests under exploration)
- Atomicity annotations for C-extension calls
- Schedule minimization (shrink a failing schedule to its racy core)

## Related work

[loom](https://github.com/tokio-rs/loom) (Rust),
[CHESS](https://www.microsoft.com/en-us/research/project/chess-find-and-reproduce-heisenbugs-in-concurrent-programs/)
(Microsoft Research),
[PCT](https://www.microsoft.com/en-us/research/publication/a-randomized-scheduler-with-probabilistic-guarantees-of-finding-bugs/)
(Burckhardt et al., ASPLOS 2010),
[rr chaos mode](https://rr-project.org/),
[pytest-run-parallel](https://github.com/Quansight-Labs/pytest-run-parallel).

## License

MIT © Danny Kissel
