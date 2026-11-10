#!/usr/bin/env python3
"""Comprehensive Latency and Scaling Benchmark for Pescador Prefetching.

This standalone script reproduces all empirical benchmarks presented in the
Pescador Concurrent Prefetching documentation:
1. Workload 1: I/O-Bound Multiplexing (concurrency speedup & stall time reduction)
2. Workload 2: Latency Jitter & Spikes (tail latency and spike absorption)
3. Workload 3: Active Stream Scaling (A in {2, 4, 8, 16, 32, 64})
4. Workload 4: Buffer Depth Sensitivity (B in {1, 2, 4, 8})

Usage:
------
    python benchmark_prefetching.py
    python benchmark_prefetching.py --workload io
    python benchmark_prefetching.py --workload scaling
    python benchmark_prefetching.py --quick
"""

import argparse
import time
import numpy as np
import pescador


def make_io_stream(stream_id, n_samples=100, io_delay_s=0.010):
    """Simulate a stream with fixed I/O latency."""
    for i in range(n_samples):
        time.sleep(io_delay_s)
        yield {
            "stream_id": stream_id,
            "sample_id": i,
            "data": np.ones((64, 64), dtype=np.float32),
        }


def make_jitter_stream(stream_id, n_samples=100, base_delay_s=0.008, spike_delay_s=0.025, spike_prob=0.10):
    """Simulate a stream with intermittent I/O spikes (e.g. cache misses, network lag)."""
    rng = np.random.RandomState(seed=stream_id + 1000)
    for i in range(n_samples):
        if rng.rand() < spike_prob:
            time.sleep(spike_delay_s)
        else:
            time.sleep(base_delay_s)
        yield {
            "stream_id": stream_id,
            "sample_id": i,
            "data": np.ones((64, 64), dtype=np.float32),
        }


def format_stats(latencies_ms):
    arr = np.array(latencies_ms)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _print_workload_table(total_sync, total_async, stats_s, stats_a, tail_p=90):
    speedup = total_sync / total_async
    mean_drop = (1 - stats_a["mean"] / stats_s["mean"]) * 100
    p50_drop = (1 - stats_a["p50"] / stats_s["p50"]) * 100
    tail_key = f"p{tail_p}"
    tail_drop = (1 - stats_a[tail_key] / stats_s[tail_key]) * 100
    p99_drop = (1 - stats_a["p99"] / stats_s["p99"]) * 100

    print(f"  {'Metric':<18} | {'Synchronous':<14} | {'Prefetched':<14} | {'Improvement'}")
    print(f"  {'-'*18}-|-{'-'*14}-|-{'-'*14}-|-{'-'*15}")
    print(f"  {'Total Wall Time':<18} | {total_sync:9.3f} s    | {total_async:9.3f} s    | {speedup:5.2f}x speedup")
    print(f"  {'Mean Stall':<18} | {stats_s['mean']:9.2f} ms   | {stats_a['mean']:9.2f} ms   | {mean_drop:5.1f}% drop")
    print(f"  {'P50 Stall':<18} | {stats_s['p50']:9.2f} ms   | {stats_a['p50']:9.2f} ms   | {p50_drop:5.1f}% drop")
    print(
        f"  {f'P{tail_p} Stall':<18} | {stats_s[tail_key]:9.2f} ms   |"
        f" {stats_a[tail_key]:9.2f} ms   | {tail_drop:5.1f}% drop"
    )
    print(f"  {'P99 Stall':<18} | {stats_s['p99']:9.2f} ms   | {stats_a['p99']:9.2f} ms   | {p99_drop:5.1f}% drop")


def run_io_bound_benchmark(n_items=250, n_active=8, buffer_size=4):
    print(f"\n{'='*75}")
    print(f"Workload 1: I/O-Bound Multiplexing (n_items={n_items}, A={n_active}, I/O=10ms, consumer=1ms)")
    print(f"{'='*75}")

    # Synchronous baseline
    s_sync = [pescador.Streamer(make_io_stream, i, n_samples=100, io_delay_s=0.010) for i in range(16)]
    mux_sync = pescador.StochasticMux(s_sync, n_active=n_active, rate=32, buffer_size=None, random_state=42)

    wait_sync = []
    t0_all = time.perf_counter()
    with mux_sync:
        it = iter(mux_sync)
        for _ in range(n_items):
            t0 = time.perf_counter()
            next(it)
            wait_sync.append((time.perf_counter() - t0) * 1000.0)
            time.sleep(0.001)
    total_sync = time.perf_counter() - t0_all

    # Prefetched async
    s_async = [pescador.Streamer(make_io_stream, i, n_samples=100, io_delay_s=0.010) for i in range(16)]
    mux_async = pescador.StochasticMux(s_async, n_active=n_active, rate=32, buffer_size=buffer_size, random_state=42)

    wait_async = []
    t0_all = time.perf_counter()
    with mux_async:
        it = iter(mux_async)
        for _ in range(n_items):
            t0 = time.perf_counter()
            next(it)
            wait_async.append((time.perf_counter() - t0) * 1000.0)
            time.sleep(0.001)
    total_async = time.perf_counter() - t0_all

    stats_s = format_stats(wait_sync)
    stats_a = format_stats(wait_async)
    _print_workload_table(total_sync, total_async, stats_s, stats_a, tail_p=90)


def run_jitter_benchmark(n_items=250, n_active=4, buffer_size=4):
    print(f"\n{'='*75}")
    print(f"Workload 2: Latency Jitter & Spikes (n_items={n_items}, A={n_active}, 90% 8ms / 10% 25ms, consumer=8ms)")
    print(f"{'='*75}")

    # Synchronous
    s_sync = [pescador.Streamer(make_jitter_stream, i, n_samples=100) for i in range(16)]
    mux_sync = pescador.StochasticMux(s_sync, n_active=n_active, rate=32, buffer_size=None, random_state=42)

    wait_sync = []
    t0_all = time.perf_counter()
    with mux_sync:
        it = iter(mux_sync)
        for _ in range(n_items):
            t0 = time.perf_counter()
            next(it)
            wait_sync.append((time.perf_counter() - t0) * 1000.0)
            time.sleep(0.008)
    total_sync = time.perf_counter() - t0_all

    # Prefetched
    s_async = [pescador.Streamer(make_jitter_stream, i, n_samples=100) for i in range(16)]
    mux_async = pescador.StochasticMux(s_async, n_active=n_active, rate=32, buffer_size=buffer_size, random_state=42)

    wait_async = []
    t0_all = time.perf_counter()
    with mux_async:
        it = iter(mux_async)
        for _ in range(n_items):
            t0 = time.perf_counter()
            next(it)
            wait_async.append((time.perf_counter() - t0) * 1000.0)
            time.sleep(0.008)
    total_async = time.perf_counter() - t0_all

    stats_s = format_stats(wait_sync)
    stats_a = format_stats(wait_async)
    _print_workload_table(total_sync, total_async, stats_s, stats_a, tail_p=95)


def run_scaling_benchmark(n_items=200, active_counts=None):
    if active_counts is None:
        active_counts = [2, 4, 8, 16, 32, 64]

    print(f"\n{'='*75}")
    print(f"Workload 3: Active Stream Scaling A in {active_counts} (n_items={n_items}, I/O=15ms, consumer=1ms)")
    print(f"{'='*75}")

    print(f"  {'A (Active)':<10} | {'Synchronous':<16} | {'Prefetched':<16} | {'Speedup'}")
    print(f"  {'-'*10}-|-{'-'*16}-|-{'-'*16}-|-{'-'*10}")

    for n_act in active_counts:
        n_pool = max(128, n_act * 2)
        s_sync = [pescador.Streamer(make_io_stream, i, n_samples=100, io_delay_s=0.015) for i in range(n_pool)]
        mux_sync = pescador.StochasticMux(s_sync, n_active=n_act, rate=32, buffer_size=None, random_state=42)
        t_start = time.perf_counter()
        with mux_sync:
            it = iter(mux_sync)
            for _ in range(n_items):
                next(it)
                time.sleep(0.001)
        t_sync = time.perf_counter() - t_start

        s_async = [pescador.Streamer(make_io_stream, i, n_samples=100, io_delay_s=0.015) for i in range(n_pool)]
        mux_async = pescador.StochasticMux(s_async, n_active=n_act, rate=32, buffer_size=4, random_state=42)
        t_start = time.perf_counter()
        with mux_async:
            it = iter(mux_async)
            for _ in range(n_items):
                next(it)
                time.sleep(0.001)
        t_async = time.perf_counter() - t_start

        print(f"  {n_act:<10} | {t_sync:11.3f} s    | {t_async:11.3f} s    | {t_sync/t_async:5.2f}x")


def run_buffer_sensitivity_benchmark(n_items=200):
    print(f"\n{'='*75}")
    print(f"Workload 4: Buffer Depth Sensitivity (n_items={n_items}, A=4, I/O=10ms, consumer=6ms)")
    print(f"{'='*75}")

    buffer_sizes = [None, 1, 2, 4, 8]
    print(f"  {'Configuration':<16} | {'Total Wall Time':<16} | {'Mean Stall':<14} | {'P50 Stall'}")
    print(f"  {'-'*16}-|-{'-'*16}-|-{'-'*14}-|-{'-'*12}")

    for bs in buffer_sizes:
        streamers = [pescador.Streamer(make_io_stream, i, n_samples=100, io_delay_s=0.010) for i in range(16)]
        mux = pescador.StochasticMux(streamers, n_active=4, rate=32, buffer_size=bs, random_state=42)

        wait_times = []
        t_start = time.perf_counter()
        with mux:
            it = iter(mux)
            for _ in range(n_items):
                t0 = time.perf_counter()
                next(it)
                wait_times.append((time.perf_counter() - t0) * 1000.0)
                time.sleep(0.006)
        total_time = time.perf_counter() - t_start

        arr = np.array(wait_times)
        label = "Sync (None)" if bs is None else f"buffer_size={bs}"
        p50 = np.percentile(arr, 50)
        print(f"  {label:<16} | {total_time:11.3f} s    | {np.mean(arr):9.2f} ms    | {p50:7.3f} ms")


def main():
    parser = argparse.ArgumentParser(description="Pescador Concurrent Prefetching Benchmarks")
    parser.add_argument("--workload", choices=["all", "io", "jitter", "scaling", "buffer"], default="all")
    parser.add_argument("--quick", action="store_true", help="Run with reduced item count for fast testing")
    args = parser.parse_args()

    n_items = 50 if args.quick else 200

    if args.workload in ("all", "io"):
        run_io_bound_benchmark(n_items=n_items)
    if args.workload in ("all", "jitter"):
        run_jitter_benchmark(n_items=n_items)
    if args.workload in ("all", "scaling"):
        active_counts = [2, 4, 8] if args.quick else [2, 4, 8, 16, 32, 64]
        run_scaling_benchmark(n_items=n_items, active_counts=active_counts)
    if args.workload in ("all", "buffer"):
        run_buffer_sensitivity_benchmark(n_items=n_items)


if __name__ == "__main__":
    main()
