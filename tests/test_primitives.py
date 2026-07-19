"""Modelled primitives behave like the real ones, including via patching."""

import queue
import threading
import time

import pytest

import interleave_test as it


def test_queue_producer_consumer():
    # queue.Queue builds its locks/conditions from patched threading names,
    # so a bounded queue exercises Lock + Condition wait/notify heavily.
    def model():
        q = queue.Queue(maxsize=2)
        seen = []

        def producer():
            for x in range(4):
                q.put(x)

        def consumer():
            for _ in range(4):
                seen.append(q.get())

        p = it.spawn(producer)
        c = it.spawn(consumer)
        p.join()
        c.join()
        assert seen == [0, 1, 2, 3]

    res = it.explore(model, iterations=40, seed=5)
    assert res.ok


def test_event_and_lock_timeout():
    def model():
        started = it.Event()
        done = it.Event()
        lock = it.Lock()

        def holder():
            with lock:
                started.set()
                done.wait()  # parks while holding the lock

        def contender():
            started.wait()
            got = lock.acquire(timeout=0.01)
            assert got is False  # holder is parked holding it
            done.set()  # let the holder finish

        t1 = it.spawn(holder)
        t2 = it.spawn(contender)
        t1.join()
        t2.join()

    res = it.explore(model, iterations=30, seed=3)
    assert res.ok


def test_semaphore_bounds_concurrency():
    def model():
        sem = it.Semaphore(1)
        active = [0]
        max_seen = [0]

        def worker():
            with sem:
                active[0] += 1
                max_seen[0] = max(max_seen[0], active[0])
                active[0] -= 1

        tasks = [it.spawn(worker) for _ in range(3)]
        for t in tasks:
            t.join()
        assert max_seen[0] == 1

    assert it.explore(model, iterations=60, seed=11).ok


def test_bounded_semaphore_over_release():
    def model():
        sem = it.BoundedSemaphore(1)
        with pytest.raises(ValueError):
            sem.release()

    assert it.explore(model, iterations=5, seed=0).ok


def test_condition_wait_notify():
    def model():
        cond = it.Condition()
        box = []

        def consumer():
            with cond:
                while not box:
                    cond.wait()
                assert box == ["item"]

        def producer():
            with cond:
                box.append("item")
                cond.notify()

        t1 = it.spawn(consumer)
        t2 = it.spawn(producer)
        t1.join()
        t2.join()

    assert it.explore(model, iterations=60, seed=13).ok


def test_rlock_reentrant():
    def model():
        lock = it.RLock()
        log = []

        def outer():
            with lock:
                with lock:  # reentrant acquire must not self-deadlock
                    log.append("in")

        t1 = it.spawn(outer)
        t2 = it.spawn(outer)
        t1.join()
        t2.join()
        assert log == ["in", "in"]

    assert it.explore(model, iterations=40, seed=17).ok


def test_patched_threading_thread_and_sleep():
    def model():
        results = []

        def work(tag):
            time.sleep(0.5)  # patched: pure scheduling point, returns instantly
            results.append(tag)

        t1 = threading.Thread(target=work, args=("a",))
        t2 = threading.Thread(target=work, args=("b",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert sorted(results) == ["a", "b"]

    started = time.monotonic()
    res = it.explore(model, iterations=30, seed=8)
    assert res.ok
    # 60 patched sleeps of 0.5s must not take 30 wall-clock seconds.
    assert time.monotonic() - started < 20


def test_patched_threading_lock_inside_model():
    def model():
        c = {"n": 0}
        lock = threading.Lock()  # patched name -> modelled lock

        def inc():
            with lock:
                c["n"] += 1

        t1 = threading.Thread(target=inc)
        t2 = threading.Thread(target=inc)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert c["n"] == 2

    assert it.explore(model, iterations=100, seed=1234).ok


def test_spawn_result():
    def model():
        t = it.spawn(lambda: 41 + 1)
        t.join()
        assert t.result() == 42

    assert it.explore(model, iterations=3, seed=0).ok
