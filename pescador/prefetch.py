#!/usr/bin/env python
"""Concurrent prefetching using asyncio TaskGroup and a hybrid thread-safe buffer."""

import asyncio
import collections
import threading
from typing import Any, Optional

from decorator import decorator

from . import core
from .exceptions import PescadorError

__all__ = [
    "HybridBuffer",
    "TaskGroupPrefetchBridge",
    "SingleStreamPrefetchBridge",
    "PrefetchStreamer",
    "prefetchable",
]

_SENTINEL = object()


class HybridBuffer:
    """A bounded FIFO buffer bridging an asyncio producer with a synchronous consumer.

    The synchronous consumer calls ``sync_get()`` which locks the internal mutex,
    checks the deque, and pops immediately (< 1 microsecond when pre-buffered).

    The asyncio producer calls ``await async_put(item)`` which appends to the deque
    under the lock, notifies waiting consumers, and if the buffer is full,
    asynchronously awaits an ``asyncio.Event`` until the consumer frees space.
    """

    def __init__(self, maxsize: int = 2, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Initialize the hybrid buffer.

        Parameters
        ----------
        maxsize : int > 0
            Maximum number of items to retain in the buffer.
        loop : asyncio.AbstractEventLoop or None
            The event loop running the producer coroutines.
        """
        if maxsize <= 0:
            raise PescadorError(f"maxsize must be > 0, got {maxsize}")

        self.maxsize = maxsize
        self.loop = loop
        self.queue: collections.deque = collections.deque()
        self.lock = threading.Lock()
        self.has_item = threading.Condition(self.lock)
        self.has_space = asyncio.Event()
        self.has_space.set()  # Initially has room
        self.closed = False
        self.exception: Optional[BaseException] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop if not provided at init."""
        self.loop = loop

    async def async_put(self, item: Any) -> None:
        """Asynchronously push an item into the buffer.

        Suspends if the buffer has reached ``maxsize`` until space is made by ``sync_get()``.

        Parameters
        ----------
        item : Any
            The item to place into the buffer.

        Raises
        ------
        asyncio.CancelledError
            If the buffer is closed while waiting for space.
        """
        while True:
            with self.lock:
                if self.closed:
                    raise asyncio.CancelledError("Buffer is closed")
                if len(self.queue) < self.maxsize:
                    self.queue.append(item)
                    self.has_item.notify()
                    return
                # Buffer is full: clear has_space while holding self.lock so that
                # await self.has_space.wait() will block until a consumer drains an item
                # and signals has_space.set via call_soon_threadsafe.
                self.has_space.clear()
            # Await space without holding the thread lock
            await self.has_space.wait()

    def sync_get(self, timeout: Optional[float] = None) -> Any:
        """Synchronously pull an item from the buffer.

        Parameters
        ----------
        timeout : float or None
            Maximum time in seconds to wait for an item. If None, wait indefinitely.

        Returns
        -------
        item : Any
            The next item from the buffer.

        Raises
        ------
        StopIteration
            If the buffer is closed and empty.
        TimeoutError
            If timeout expires before an item is available.
        Exception
            If an exception was captured by the producer task.
        """
        with self.has_item:
            while not self.queue:
                if self.exception is not None:
                    raise self.exception
                if self.closed:
                    raise StopIteration
                if not self.has_item.wait(timeout=timeout):
                    raise TimeoutError(f"sync_get timed out after {timeout} seconds")

            item = self.queue.popleft()
            # If the loop is running, signal that space is now available
            if self.loop is not None and not self.loop.is_closed():
                self.loop.call_soon_threadsafe(self.has_space.set)
            return item

    def close(self) -> None:
        """Close the buffer, waking any waiting producers and consumers."""
        with self.lock:
            self.closed = True
            self.has_item.notify_all()
            if self.loop is not None and not self.loop.is_closed():
                self.loop.call_soon_threadsafe(self.has_space.set)

    def set_exception(self, exc: BaseException) -> None:
        """Capture an exception and wake any waiting consumer to raise it."""
        with self.lock:
            self.exception = exc
            self.closed = True
            self.has_item.notify_all()
            if self.loop is not None and not self.loop.is_closed():
                self.loop.call_soon_threadsafe(self.has_space.set)

    def is_empty(self) -> bool:
        """Return True if the queue currently holds no items."""
        with self.lock:
            return len(self.queue) == 0

    def qsize(self) -> int:
        """Return the current number of items in the queue."""
        with self.lock:
            return len(self.queue)


class TaskGroupPrefetchBridge:
    """Manages an asyncio TaskGroup on a background thread for concurrent stream prefetching."""

    def __init__(self, mux: Any, buffer_size: int = 2, timeout: Optional[float] = 5.0):
        """Initialize the prefetch bridge.

        Parameters
        ----------
        mux : BaseMux
            The mux instance whose active streams will be prefetched.
        buffer_size : int > 0
            The buffer size per active stream.
        timeout : float or None
            Timeout for thread shutdown and queue operations.
        """
        self.mux = mux
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self.stop_event: Optional[asyncio.Event] = None
        self.buffers: list[HybridBuffer] = []
        self._ready_event = threading.Event()

    def start(self) -> None:
        """Start the background thread and initialize prefetch tasks."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=self.timeout)

    def _run_loop(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        n_slots = len(self.mux.streams_)
        self.buffers = [HybridBuffer(maxsize=self.buffer_size, loop=self.loop) for _ in range(n_slots)]

        async def main():
            self.stop_event = asyncio.Event()
            self._ready_event.set()
            try:
                async with asyncio.TaskGroup() as tg:
                    for slot_idx in range(n_slots):
                        tg.create_task(self._slot_worker(slot_idx, self.buffers[slot_idx]))
                    await self.stop_event.wait()
            except* asyncio.CancelledError:
                pass
            except* Exception as eg:
                for exc in eg.exceptions:
                    for buf in self.buffers:
                        buf.set_exception(exc)

        try:
            self.loop.run_until_complete(main())
        finally:
            self.loop.close()

    async def _slot_worker(self, slot_idx: int, buffer: HybridBuffer) -> None:
        """Worker task prefetching samples for a single active stream slot."""
        try:
            consecutive_exhaustions = 0
            n_streams_total = getattr(self.mux, "n_streams", None) or (
                len(self.mux.streams_) if self.mux.streams_ else 10
            )
            max_consecutive = max(100, 10 * n_streams_total)
            while not self.stop_event.is_set():
                with self.mux._state_lock:
                    stream = self.mux.streams_[slot_idx] if slot_idx < len(self.mux.streams_) else None

                if stream is None:
                    break

                # Fetch next sample from the streamer
                try:
                    if hasattr(stream, "__anext__"):
                        item = await stream.__anext__()
                    else:
                        item = await asyncio.to_thread(next, stream, _SENTINEL)
                except StopAsyncIteration:
                    item = _SENTINEL
                except Exception as exc:
                    buffer.set_exception(exc)
                    raise

                if item is _SENTINEL:
                    consecutive_exhaustions += 1
                    with self.mux._state_lock:
                        old_stream = self.mux.streams_[slot_idx] if slot_idx < len(self.mux.streams_) else None
                        saved_weight = (
                            self.mux.stream_weights_[slot_idx]
                            if hasattr(self.mux, "stream_weights_")
                            else None
                        )
                        self.mux._on_stream_exhausted(slot_idx)
                        self.mux._replace_stream(slot_idx)
                        new_stream = self.mux.streams_[slot_idx] if slot_idx < len(self.mux.streams_) else None

                        if (
                            new_stream is None
                            or new_stream is old_stream
                            or not self.mux._streamers_available()
                            or (hasattr(self.mux, "stream_weights_") and self.mux.stream_weights_[slot_idx] == 0.0)
                            or consecutive_exhaustions >= max_consecutive
                        ):
                            # Restore stream pointer and weight if needed so consumer can drain buffered items
                            if not buffer.is_empty():
                                if slot_idx < len(self.mux.streams_):
                                    self.mux.streams_[slot_idx] = old_stream
                                if saved_weight is not None:
                                    import numpy as np
                                    self.mux.stream_weights_[slot_idx] = saved_weight
                                    self.mux.weight_norm_ = np.sum(self.mux.stream_weights_)
                            break
                    continue

                consecutive_exhaustions = 0

                await buffer.async_put(item)

                with self.mux._state_lock:
                    self.mux.stream_counts_[slot_idx] += 1

        except asyncio.CancelledError:
            pass
        finally:
            buffer.close()

    def pull(self, slot_idx: int, timeout: Optional[float] = None) -> Any:
        """Synchronously pull the next pre-buffered item from slot_idx.

        Parameters
        ----------
        slot_idx : int
            The active stream index chosen by the mux.
        timeout : float or None
            Maximum time to wait.

        Returns
        -------
        sample : Any
            The sample yielded by the active stream.
        """
        return self.buffers[slot_idx].sync_get(timeout=timeout)

    def close(self) -> None:
        """Shut down the prefetch bridge and wait for background tasks to terminate."""
        if self.loop is not None and not self.loop.is_closed():
            if self.stop_event is not None:
                self.loop.call_soon_threadsafe(self.stop_event.set)
        for buf in self.buffers:
            buf.close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self.timeout)


class SingleStreamPrefetchBridge:
    """Manages an asyncio TaskGroup on a background thread for a single stream."""

    def __init__(self, stream: Any, buffer_size: int = 2, timeout: Optional[float] = 5.0):
        self.stream = stream
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self.stop_event: Optional[asyncio.Event] = None
        self.buffer: Optional[HybridBuffer] = None
        self._ready_event = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready_event.wait(timeout=self.timeout)

    def _run_loop(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.buffer = HybridBuffer(maxsize=self.buffer_size, loop=self.loop)

        async def main():
            self.stop_event = asyncio.Event()
            self._ready_event.set()
            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._worker())
                    await self.stop_event.wait()
            except* asyncio.CancelledError:
                pass
            except* Exception as eg:
                for exc in eg.exceptions:
                    self.buffer.set_exception(exc)

        try:
            self.loop.run_until_complete(main())
        finally:
            self.loop.close()

    async def _worker(self) -> None:
        try:
            while not self.stop_event.is_set():
                try:
                    if hasattr(self.stream, "__anext__"):
                        item = await self.stream.__anext__()
                    else:
                        item = await asyncio.to_thread(next, self.stream, _SENTINEL)
                except StopAsyncIteration:
                    item = _SENTINEL
                except Exception as exc:
                    self.buffer.set_exception(exc)
                    raise

                if item is _SENTINEL:
                    break

                await self.buffer.async_put(item)
        except asyncio.CancelledError:
            pass
        finally:
            self.buffer.close()

    def pull(self, timeout: Optional[float] = None) -> Any:
        return self.buffer.sync_get(timeout=timeout)

    def close(self) -> None:
        if self.loop is not None and not self.loop.is_closed():
            if self.stop_event is not None:
                self.loop.call_soon_threadsafe(self.stop_event.set)
        if self.buffer is not None:
            self.buffer.close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self.timeout)


class PrefetchStreamer(core.Streamer):
    """Wraps any streamer to prefetch samples asynchronously into a hybrid buffer.

    Parameters
    ----------
    streamer : iterable or generator function
        Any generator function or object that is iterable when instantiated.
    buffer_size : int > 0, default=2
        Number of items to prefetch ahead of the consumer.
    timeout : float, default=5.0
        Timeout in seconds for background task termination.
    *args, **kwargs
        Additional arguments passed to ``streamer()`` if callable.

    Examples
    --------
    >>> def slow_gen():
    ...     for i in range(5):
    ...         yield i
    >>> S = pescador.Streamer(slow_gen)
    >>> P = pescador.PrefetchStreamer(S, buffer_size=3)
    >>> for item in P:
    ...     print(item)
    0
    1
    2
    3
    4
    """

    def __init__(self, streamer: Any, buffer_size: int = 2, timeout: float = 5.0, *args, **kwargs):
        super().__init__(streamer, *args, **kwargs)
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.bridge_ = None

    def __deepcopy__(self, memo):
        import copy
        cls = self.__class__
        copy_result = cls.__new__(cls)
        memo[id(self)] = copy_result
        for k, v in self.__dict__.items():
            if k in ("bridge_", "_active_copy"):
                setattr(copy_result, k, None)
            else:
                try:
                    setattr(copy_result, k, copy.deepcopy(v, memo))
                except TypeError:
                    setattr(copy_result, k, v)
        return copy_result

    def __getstate__(self):
        """Prepare PrefetchStreamer for pickling / multiprocessing serialization."""
        state = self.__dict__.copy()
        state["bridge_"] = None
        state["_active_copy"] = None
        return state

    def __setstate__(self, state):
        """Restore PrefetchStreamer from pickled state."""
        self.__dict__.update(state)
        self.bridge_ = None
        self._active_copy = None

    def __enter__(self, *args, **kwargs):
        streamer_copy = super().__enter__(*args, **kwargs)
        if streamer_copy.is_activated_copy and streamer_copy.bridge_ is None:
            streamer_copy.bridge_ = SingleStreamPrefetchBridge(
                streamer_copy.stream_, buffer_size=self.buffer_size, timeout=self.timeout
            )
            streamer_copy.bridge_.start()
            self._active_copy = streamer_copy
        return streamer_copy

    def __exit__(self, *exc):
        if hasattr(self, "_active_copy") and self._active_copy is not None:
            if hasattr(self._active_copy, "bridge_") and self._active_copy.bridge_ is not None:
                self._active_copy.bridge_.close()
                self._active_copy.bridge_ = None
            self._active_copy = None
        if hasattr(self, "bridge_") and self.bridge_ is not None:
            self.bridge_.close()
            self.bridge_ = None
        return super().__exit__(*exc)

    def iterate(self, max_iter: Optional[int] = None):
        with self as active:
            n = 0
            while max_iter is None or n < max_iter:
                try:
                    yield active.bridge_.pull()
                    n += 1
                except StopIteration:
                    break


@decorator
def prefetchable(function, buffer_size=2, timeout=None, *args, **kwargs):
    """Create a PrefetchStreamer object by decoration.

    This decorator marks a generator function as prefetchable, wrapping it
    in a `Streamer` and `PrefetchStreamer` automatically upon invocation.

    Parameters
    ----------
    function : callable
        The generator function to decorate.
    buffer_size : int > 0, default=2
        Number of items to prefetch asynchronously.
    timeout : float > 0 or None, default=None
        Maximum time (seconds) to wait before raising TimeoutError in consumer.

    Examples
    --------
    >>> @pescador.prefetchable
    ... def myfunc(n):
    ...     for i in range(n):
    ...         yield i
    >>> s = myfunc(5)

    With customized buffer depth and timeout:

    >>> @pescador.prefetchable(buffer_size=4, timeout=5.0)
    ... def myfunc_buffered(n):
    ...     for i in range(n):
    ...         yield i
    >>> s = myfunc_buffered(5)
    """
    return PrefetchStreamer(
        core.Streamer(function, *args, **kwargs),
        buffer_size=buffer_size,
        timeout=timeout,
    )
