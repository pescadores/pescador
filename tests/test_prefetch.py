import time
import pytest
import numpy as np

import pescador
from pescador.prefetch import HybridBuffer, PrefetchStreamer


def test_hybrid_buffer_basic():
    buf = HybridBuffer(maxsize=3)
    assert buf.is_empty()
    assert buf.qsize() == 0

    # Test error when maxsize <= 0
    with pytest.raises(pescador.PescadorError):
        HybridBuffer(maxsize=0)


def test_hybrid_buffer_close():
    buf = HybridBuffer(maxsize=2)
    buf.close()
    assert buf.closed
    with pytest.raises(StopIteration):
        buf.sync_get()


def test_hybrid_buffer_exception():
    buf = HybridBuffer(maxsize=2)
    buf.set_exception(ValueError("test error"))
    with pytest.raises(ValueError, match="test error"):
        buf.sync_get()


def test_prefetch_streamer_finite():
    def gen():
        for i in range(10):
            yield i

    streamer = pescador.Streamer(gen)
    prefetched = PrefetchStreamer(streamer, buffer_size=3)

    items = list(prefetched)
    assert items == list(range(10))


def test_prefetch_streamer_max_iter():
    def gen():
        for i in range(100):
            yield i

    streamer = pescador.Streamer(gen)
    prefetched = PrefetchStreamer(streamer, buffer_size=3)

    items = list(prefetched.iterate(max_iter=7))
    assert items == list(range(7))


def test_prefetch_streamer_early_break():
    def gen():
        for i in range(1000):
            yield i

    streamer = pescador.Streamer(gen)
    prefetched = PrefetchStreamer(streamer, buffer_size=3)

    collected = []
    for item in prefetched:
        collected.append(item)
        if len(collected) == 5:
            break
    assert collected == list(range(5))


def test_prefetch_streamer_cycle():
    def gen():
        for i in range(3):
            yield i

    streamer = pescador.Streamer(gen)
    prefetched = PrefetchStreamer(streamer, buffer_size=2)

    collected = []
    for item in prefetched.cycle(max_iter=8):
        collected.append(item)
    assert collected == [0, 1, 2, 0, 1, 2, 0, 1]


def test_prefetch_streamer_exception():
    def gen():
        yield 1
        yield 2
        raise RuntimeError("boom")

    streamer = pescador.Streamer(gen)
    prefetched = PrefetchStreamer(streamer, buffer_size=2)

    with pytest.raises(RuntimeError, match="boom"):
        list(prefetched)


def test_stochastic_mux_prefetched_finite():
    a = pescador.Streamer("abc")
    b = pescador.Streamer("def")

    mux = pescador.StochasticMux(
        [a, b], n_active=2, rate=None, mode="exhaustive", random_state=42, buffer_size=2
    )

    items = "".join(list(mux))
    assert len(items) == 6
    assert set(items) == set("abcdef")


def test_stochastic_mux_prefetched_with_replacement():
    a = pescador.Streamer("a")
    b = pescador.Streamer("b")

    mux = pescador.StochasticMux(
        [a, b], n_active=2, rate=3, mode="with_replacement", random_state=42, buffer_size=2
    )

    items = list(mux(max_iter=20))
    assert len(items) == 20
    assert set(items) == {"a", "b"}


def test_stochastic_mux_prefetched_single_active():
    a = pescador.Streamer("a")
    b = pescador.Streamer("b")

    mux = pescador.StochasticMux(
        [a, b], n_active=2, rate=4, mode="single_active", random_state=42, buffer_size=2
    )

    items = list(mux(max_iter=20))
    assert len(items) == 20
    assert set(items) == {"a", "b"}


def test_shuffled_mux_prefetched():
    a = pescador.Streamer("a")
    b = pescador.Streamer("b")
    c = pescador.Streamer("c")

    mux = pescador.ShuffledMux([a, b, c], random_state=42, buffer_size=2)
    items = list(mux(max_iter=15))
    assert len(items) == 15
    assert set(items) == {"a", "b", "c"}


def test_round_robin_mux_prefetched():
    a = pescador.Streamer("abc")
    b = pescador.Streamer("123")

    mux = pescador.RoundRobinMux([a, b], mode="exhaustive", buffer_size=2)
    items = "".join(list(mux))
    assert items == "a1b2c3"


def test_chain_mux_prefetched():
    a = pescador.Streamer("abc")
    b = pescador.Streamer("def")

    mux = pescador.ChainMux([a, b], mode="exhaustive", buffer_size=2)
    items = "".join(list(mux))
    assert items == "abcdef"


def test_prefetched_concurrency_speedup():
    """Verify that multiple slow streams run concurrently when prefetched."""
    def slow_stream(label, count=4, delay=0.04):
        for i in range(count):
            time.sleep(delay)
            yield f"{label}_{i}"

    s1 = pescador.Streamer(slow_stream, "A", count=4, delay=0.04)
    s2 = pescador.Streamer(slow_stream, "B", count=4, delay=0.04)

    # With prefetching: 2 streams producing 4 items each (total 8 items)
    # Sequential time would be 8 * 0.04 = 0.32s
    # Concurrent prefetching should complete significantly faster (< 0.25s)
    start = time.perf_counter()
    mux = pescador.RoundRobinMux([s1, s2], mode="exhaustive", buffer_size=3)
    items = list(mux)
    elapsed = time.perf_counter() - start

    assert len(items) == 8
    assert elapsed < 0.26, f"Expected concurrent speedup, took {elapsed:.3f}s"


def test_prefetched_with_buffer_stream():
    def data_gen(val):
        for i in range(10):
            yield dict(X=np.array([val]))

    s1 = pescador.Streamer(data_gen, 1)
    s2 = pescador.Streamer(data_gen, 2)

    mux = pescador.ShuffledMux([s1, s2], random_state=42, buffer_size=2)
    buffered = pescador.buffer_stream(mux, buffer_size=4)

    batches = []
    for batch in buffered:
        batches.append(batch)
        if len(batches) == 3:
            break

    assert len(batches) == 3
    for b in batches:
        assert "X" in b
        assert b["X"].shape == (4, 1)


def _zmq_data_gen(val):
    for i in range(10):
        yield dict(X=np.array([val, i]))


def test_prefetched_with_zmq_stream():
    s1 = pescador.Streamer(_zmq_data_gen, 1)
    s2 = pescador.Streamer(_zmq_data_gen, 2)

    mux = pescador.RoundRobinMux([s1, s2], mode="exhaustive", buffer_size=2)
    zmq_stream = pescador.ZMQStreamer(mux)

    items = list(zmq_stream)
    assert len(items) == 20
    assert all("X" in it for it in items)


def test_prefetched_pickle_serialization():
    import pickle

    s1 = pescador.Streamer(_zmq_data_gen, 1)
    s2 = pescador.Streamer(_zmq_data_gen, 2)
    mux = pescador.RoundRobinMux([s1, s2], mode="exhaustive", buffer_size=2)

    unpickled_mux = pickle.loads(pickle.dumps(mux))
    assert unpickled_mux.buffer_size == 2
    items = list(unpickled_mux)
    assert len(items) == 20

    ps = pescador.PrefetchStreamer(s1, buffer_size=3)
    unpickled_ps = pickle.loads(pickle.dumps(ps))
    assert unpickled_ps.buffer_size == 3
    items_ps = list(unpickled_ps)
    assert len(items_ps) == 10
