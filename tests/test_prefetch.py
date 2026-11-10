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
    def slow_stream(label, count=4, delay=0.05):
        for i in range(count):
            time.sleep(delay)
            yield f"{label}_{i}"

    s1 = pescador.Streamer(slow_stream, "A", count=4, delay=0.05)
    s2 = pescador.Streamer(slow_stream, "B", count=4, delay=0.05)

    # Measure sequential baseline
    start_seq = time.perf_counter()
    mux_seq = pescador.RoundRobinMux([s1, s2], mode="exhaustive", buffer_size=None)
    items_seq = list(mux_seq)
    elapsed_seq = time.perf_counter() - start_seq

    # Measure concurrent prefetched
    start_pref = time.perf_counter()
    mux_pref = pescador.RoundRobinMux([s1, s2], mode="exhaustive", buffer_size=3)
    items_pref = list(mux_pref)
    elapsed_pref = time.perf_counter() - start_pref

    assert len(items_pref) == 8
    assert items_pref == items_seq
    # Concurrently prefetched should be significantly faster than sequential
    assert elapsed_pref < elapsed_seq * 0.9, (
        f"Expected concurrent speedup: prefetched ({elapsed_pref:.3f}s) "
        f"vs sequential ({elapsed_seq:.3f}s)"
    )


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


def test_hybrid_buffer_timeout():
    buf = HybridBuffer(maxsize=2)
    with pytest.raises(TimeoutError, match="sync_get timed out"):
        buf.sync_get(timeout=0.01)


def test_hybrid_buffer_set_loop():
    import asyncio

    buf = HybridBuffer(maxsize=2)
    loop = asyncio.new_event_loop()
    try:
        buf.set_loop(loop)
        assert buf.loop is loop
    finally:
        loop.close()


def test_hybrid_buffer_async_put_closed():
    import asyncio

    async def _runner():
        buf = HybridBuffer(maxsize=2)
        buf.set_loop(asyncio.get_running_loop())
        buf.close()
        with pytest.raises(asyncio.CancelledError, match="Buffer is closed"):
            await buf.async_put(123)

    asyncio.run(_runner())


def test_prefetch_streamer_async_generator():
    async def async_gen():
        for i in range(5):
            yield i

    # Test passing callable async generator function
    ps = PrefetchStreamer(async_gen, buffer_size=2)
    items = list(ps)
    assert items == list(range(5))

    # Test passing non-callable async generator instance (AsyncIterable)
    ps_inst = PrefetchStreamer(async_gen(), buffer_size=2)
    items_inst = list(ps_inst)
    assert items_inst == list(range(5))


def test_taskgroup_bridge_async_generator():
    class AsyncStreamer(pescador.Streamer):
        def __init__(self, count):
            self.count = count
            self.stream_ = None
            self.active_count_ = 0

        def _activate(self):
            async def _agen():
                for i in range(self.count):
                    yield i

            self.stream_ = _agen()

        def iterate(self, max_iter=None):
            self._activate()
            return self.stream_

    s1 = AsyncStreamer(3)
    s2 = AsyncStreamer(3)
    mux = pescador.RoundRobinMux([s1, s2], mode="exhaustive", buffer_size=2)
    items = list(mux)
    assert items == [0, 0, 1, 1, 2, 2]


def test_taskgroup_bridge_worker_exception():
    def failing_stream():
        yield 1
        raise RuntimeError("worker exception test")

    mux = pescador.RoundRobinMux([pescador.Streamer(failing_stream)], mode="exhaustive", buffer_size=2)
    with pytest.raises(RuntimeError, match="worker exception test"):
        list(mux)


def test_context_manager_bridge_cleanup():
    def count_stream():
        i = 0
        while True:
            yield i
            i += 1

    s1 = pescador.Streamer(count_stream)
    s2 = pescador.Streamer(count_stream)
    mux = pescador.RoundRobinMux([s1, s2], buffer_size=2)

    with mux as active:
        assert active.bridge_ is not None
        items = [next(active.iterate()) for _ in range(3)]
        assert len(items) == 3

    assert active.bridge_ is None

    ps = pescador.PrefetchStreamer(s1, buffer_size=2)
    with ps as active_ps:
        assert active_ps.bridge_ is not None
        items_ps = [next(active_ps.iterate()) for _ in range(3)]
        assert len(items_ps) == 3

    assert active_ps.bridge_ is None


def test_buffers_available_fallback():
    s1 = pescador.Streamer("abc")
    mux = pescador.StochasticMux([s1], n_active=1, rate=None, mode="exhaustive", buffer_size=None)
    with mux as active:
        assert active.bridge_ is None
        assert active._buffers_available()


def _make_numbered_pair(val):
    yield val
    yield val * 10


def test_stochastic_mux_prefetched_stream_replacement():
    streamers = [pescador.Streamer(_make_numbered_pair, i) for i in range(4)]
    mux = pescador.StochasticMux(
        streamers, n_active=2, rate=None, mode="exhaustive", buffer_size=2, random_state=42
    )
    items = list(mux)
    assert len(items) == 8


def test_shuffled_mux_prefetched_empty():
    a = pescador.Streamer([])
    b = pescador.Streamer([])
    mux = pescador.ShuffledMux([a, b], random_state=42, buffer_size=2)
    items = list(mux)
    assert items == []


def test_chain_mux_invalid_mode():
    with pytest.raises(pescador.PescadorError, match="Invalid ChainMux mode"):
        pescador.ChainMux([pescador.Streamer("a")], mode="invalid_mode")


def test_stochastic_mux_distribution_mismatch(monkeypatch):
    s = [pescador.Streamer("a")]
    mux = pescador.StochasticMux(s, n_active=1, rate=10)
    monkeypatch.setattr(type(mux), "n_streams", property(lambda self: 2))
    with pytest.raises(pescador.PescadorError, match="same length as `distribution`"):
        mux._activate()


def test_direct_iteration_lifecycle():
    import threading
    import time

    def gen():
        for x in range(20):
            yield x

    s1 = pescador.Streamer(gen)
    s2 = pescador.Streamer(gen)
    mux = pescador.RoundRobinMux([s1, s2], buffer_size=2)
    ps = pescador.PrefetchStreamer(gen, buffer_size=2)

    base_threads = threading.active_count()

    def _wait_for_threads(expected_count, timeout=1.0):
        start = time.perf_counter()
        while threading.active_count() > expected_count:
            if time.perf_counter() - start > timeout:
                break
            time.sleep(0.01)
        return threading.active_count()

    # 1. Clean exhaustion
    items_mux = list(mux)
    assert len(items_mux) == 40
    assert _wait_for_threads(base_threads) == base_threads

    items_ps = list(ps)
    assert len(items_ps) == 20
    assert _wait_for_threads(base_threads) == base_threads

    # 2. Early break without context manager
    for i, _ in enumerate(mux):
        if i == 3:
            break
    assert _wait_for_threads(base_threads) == base_threads

    for i, _ in enumerate(ps):
        if i == 3:
            break
    assert _wait_for_threads(base_threads) == base_threads

    # 3. Exception in consumer loop without context manager
    with pytest.raises(RuntimeError, match="consumer fail"):
        for i, _ in enumerate(mux):
            if i == 3:
                raise RuntimeError("consumer fail")
    assert _wait_for_threads(base_threads) == base_threads

    with pytest.raises(RuntimeError, match="consumer fail"):
        for i, _ in enumerate(ps):
            if i == 3:
                raise RuntimeError("consumer fail")
    assert _wait_for_threads(base_threads) == base_threads


def test_prefetchable_decorator():
    @pescador.prefetchable
    def gen_default(n, offset=0):
        for i in range(n):
            yield i + offset

    stream1 = gen_default(5, offset=10)
    assert isinstance(stream1, pescador.PrefetchStreamer)
    assert stream1.buffer_size == 2
    assert stream1.timeout is None
    assert list(stream1) == [10, 11, 12, 13, 14]

    @pescador.prefetchable(buffer_size=4, timeout=2.5)
    def gen_custom(n):
        for i in range(n):
            yield i * 2

    stream2 = gen_custom(4)
    assert isinstance(stream2, pescador.PrefetchStreamer)
    assert stream2.buffer_size == 4
    assert stream2.timeout == 2.5
    assert list(stream2) == [0, 2, 4, 6]


def test_streamable_decorator_prefetch():
    @pescador.streamable
    def gen_plain(n):
        for i in range(n):
            yield i

    s_plain = gen_plain(3)
    assert isinstance(s_plain, pescador.Streamer)
    assert not isinstance(s_plain, pescador.PrefetchStreamer)
    assert list(s_plain) == [0, 1, 2]

    @pescador.streamable(buffer_size=3, timeout=1.0)
    def gen_buffered(n):
        for i in range(n):
            yield i

    s_buf = gen_buffered(3)
    assert isinstance(s_buf, pescador.PrefetchStreamer)
    assert s_buf.buffer_size == 3
    assert s_buf.timeout == 1.0
    assert list(s_buf) == [0, 1, 2]
