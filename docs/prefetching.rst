.. _prefetching:

Concurrent Prefetching and Asynchronous Multiplexing
====================================================

Modern machine learning workloads frequently bottleneck on data ingestion.
When training or evaluating models on high-dimensional data (such as audio waveforms,
spectrograms, or images), samples must typically be fetched from storage, decompressed,
augmented, and converted to tensors before being consumed by a GPU training loop.

Historically, Pescador relied on synchronous, pull-based Python generators.
While simple and memory-efficient, this synchronous model forces the consumer thread
to block on every sample request while data is prepared.
Furthermore, in a multiplexer like :class:`~pescador.StochasticMux` or :class:`~pescador.ShuffledMux`, even when multiple
streamers are configured as "active", traditional generators execute strictly in serial:
the multiplexer selects one streamer and invokes ``next()``, pausing until that single
generator yields.

Beginning in version 3.1, Pescador introduces **concurrent prefetching** via structured
concurrency (:py:mod:`asyncio.TaskGroup`) and lock-free thread-safe queues.
This section details the theoretical latency model, architecture, empirical performance,
and guidelines for proper use of asynchronous multiplexing.


The Synchronous Bottleneck
--------------------------

To understand why prefetching is necessary, consider a typical generator pipeline consisting
of a *producer* (the generator loading and preparing data) and a *consumer* (such as a model
forward pass or batch accumulator).

Let the time required to produce a sample be a random variable :math:`T_{\text{prod}}` with mean
:math:`\mu_p`, and the time required for the consumer to process the sample be :math:`T_{\text{cons}}`
with mean :math:`\mu_c`.

In a synchronous generator pipeline:

.. math::

   T_{\text{cycle}} = T_{\text{prod}} + T_{\text{cons}}.

The expected total cycle time per sample is:

.. math::

   \text{E}[T_{\text{cycle}}] = \mu_p + \mu_c,

yielding an effective throughput of:

.. math::

   R_{\text{sync}} = \frac{1}{\mu_p + \mu_c}.

Because production and consumption are serialized on the consumer thread, the consumer is
forced to stall for :math:`T_{\text{prod}}` on every sample.
If loading an audio file takes :math:`15\,\text{ms}` and model execution takes :math:`10\,\text{ms}`,
each step takes :math:`25\,\text{ms}`, leaving the GPU idle for :math:`60\%` of the total runtime.


Theoretical Analysis of Prefetching
-----------------------------------

Overlapped Execution (Single Stream)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If data production is moved into a concurrent background task and buffered in a thread-safe
queue of capacity :math:`B \ge 1`, production and consumption proceed in parallel.
The time required to complete one cycle is bounded by the slower of the two operations:

.. math::

   T_{\text{cycle, async}} \approx \max(T_{\text{prod}}, T_{\text{cons}}).

The expected throughput increases to:

.. math::

   R_{\text{async}} = \frac{1}{\max(\mu_p, \mu_c)}.

The theoretical speedup :math:`S` over the synchronous baseline is:

.. math::

   S = \frac{R_{\text{async}}}{R_{\text{sync}}} = \frac{\mu_p + \mu_c}{\max(\mu_p, \mu_c)}.

When the producer and consumer are evenly balanced (:math:`\mu_p \approx \mu_c`),
the theoretical speedup approaches **:math:`2\times`** (a :math:`50\%` reduction in total wall-clock time).
Under this condition, the consumer's perceived stall time per sample drops from :math:`\mu_p`
down to the microscopic time required to pop an item from memory (:math:`< 1\,\mu\text{s}`).


Active Set Concurrency in Multiplexers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The benefits of prefetching are even more pronounced in multiplexers (:class:`~pescador.StochasticMux`,
:class:`~pescador.ShuffledMux`, :class:`~pescador.RoundRobinMux`, and :class:`~pescador.ChainMux`).

Consider a multiplexer with an active set of size :math:`A` (e.g., :math:`A = 8` streams).
In the synchronous implementation:

* Even though :math:`A` streams are conceptually "active", only **one** stream is polled at a time.
* The remaining :math:`A - 1` active streams sit completely idle until randomly chosen.
* Concurrency is strictly :math:`1`, regardless of whether :math:`A = 2` or :math:`A = 64`.

When concurrent prefetching is enabled with buffer depth :math:`B`:

* An internal ``asyncio.TaskGroup`` runs :math:`A` concurrent background worker tasks—one for
  each active stream slot.
* While the consumer processes sample :math:`k`, **all :math:`A` active streams are simultaneously
  loading their next samples** in the background.
* The aggregate production capacity of the active set scales linearly:

.. math::

   \lambda_{\text{prod}} = \frac{A}{\mu_p}.

The resulting system throughput becomes:

.. math::

   R_{\text{mux}} = \min\left( \frac{A}{\mu_p}, \frac{1}{\mu_c} \right).

In I/O-bound regimes where :math:`\mu_p \gg \mu_c`, the speedup over the synchronous baseline
approaches :math:`A`:

.. math::

   S_{\text{mux}} = \frac{\mu_p + \mu_c}{\frac{\mu_p}{A} + \mu_c} \approx A.

As shown empirically below, scaling :math:`A` from 2 to 16 yields near-linear throughput speedups
(:math:`2.0\times \rightarrow 9.4\times`).


Jitter Absorption and Tail Latency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Real-world storage and network I/O exhibit variance: operating system disk caches, network
congestion, and garbage collection introduce intermittent latency spikes.
Let :math:`T_{\text{prod}}` be a random variable with high variance :math:`\sigma_p^2` or occasional
extreme outliers (e.g., :math:`90\%` normal latency :math:`8\,\text{ms}`, :math:`10\%` spike :math:`25\,\text{ms}`).

In synchronous mode, every individual spike directly propagates to the consumer thread as a
pipeline stall: the 99th percentile consumer latency :math:`P_{99}` reflects the full severity
of the I/O spike.

With a prefetch buffer of capacity :math:`B \ge 2`, the buffer acts as an elastic reservoir.
Occasional slow fetches are absorbed by draining previously buffered items, while bursts of
consumer activity are cushioned by background producers.
Consequently, tail latency (:math:`P_{95}`, :math:`P_{99}`) drops by more than **:math:`99\%`**.


Architecture and Design
-----------------------

Pescador implements prefetching through an asynchronous bridge that connects synchronous
consumer threads to concurrent background producers:

.. code-block:: text

   +-----------------------------------------------------------------+
   | Consumer Thread (Main / Training Loop)                          |
   |                                                                 |
   |   for sample in mux.iterate():   <-- sync_get() (< 1 us)        |
   +-----------------------------------------------------------------+
                         |                          ^
                    acquire lock                notify item
                         v                          |
   +-----------------------------------------------------------------+
   | HybridBuffer (Bounded Mutex-Protected Deque)                    |
   |                                                                 |
   |   deque: [ item_0, item_1, ... ] (maxlen = buffer_size)         |
   |   threading.Lock  +  asyncio.Event (space available)            |
   +-----------------------------------------------------------------+
                         ^                          |
                    notify space               await space
                         |                          v
   +-----------------------------------------------------------------+
   | Background Daemon Thread (Dedicated asyncio Event Loop)         |
   |                                                                 |
   |   asyncio.TaskGroup:                                            |
   |     Worker 0 -> Stream 0 (asyncio.to_thread(next))              |
   |     Worker 1 -> Stream 1 (asyncio.to_thread(next))              |
   |     ...                                                         |
   |     Worker A-1 -> Stream A-1                                    |
   +-----------------------------------------------------------------+

Hybrid Thread-Safe Buffer (``HybridBuffer``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Standard ``asyncio.Queue`` requires an asynchronous consumer (``await queue.get()``), while
standard ``queue.Queue`` incurs heavy thread-switching overhead.
Pescador addresses this with ``HybridBuffer``:

* **Synchronous Consumer**: Calls ``sync_get()``, acquiring a lightweight ``threading.Lock``,
  popping from a ``collections.deque`` in :math:`< 1\,\mu\text{s}`, and notifying the background
  event loop via ``loop.call_soon_threadsafe(has_space.set)``.
* **Asynchronous Producer**: Calls ``await async_put(item)``. If the buffer is full
  (``len >= buffer_size``), it awaits an ``asyncio.Event`` without blocking other background tasks.
* **Bounded Memory**: Bounded by ``buffer_size``, strictly preventing runaway memory growth.

Structured Concurrency with ``asyncio.TaskGroup``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
All worker coroutines are managed within an :py:class:`asyncio.TaskGroup` inside an isolated
background thread:

* **Deterministic Teardown**: When the multiplexer exits or encounters an exception, the
  ``TaskGroup`` cleanly cancels all active worker tasks, awaits their termination, and shuts down
  the event loop.
* **Transparent Stream Replacement**: When an active streamer in ``StochasticMux`` exhausts,
  its background task automatically activates the next replacement streamer while remaining
  buffered items from the retiring streamer are drained without stalls.


Empirical Benchmarks
--------------------

The following benchmarks were conducted on Linux using Python 3.13, comparing synchronous
baseline Pescador (``buffer_size=None``) against async prefetching (``buffer_size > 0``).

Workload 1: I/O-Bound Multiplexing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Multiplexing 16 streams with :math:`10\,\text{ms}` simulated I/O latency, :math:`A = 8` active
streams, and a :math:`1\,\text{ms}` consumer:

.. csv-table:: I/O-Bound Multiplexing Results (250 items, :math:`A=8`)
   :header: "Metric", "Synchronous Baseline", "Prefetched (``buffer_size=4``)", "Reduction / Gain"
   :widths: 25, 25, 25, 25

   "**Total Wall-Clock Time**", ":math:`2.891\,\text{s}`", ":math:`0.479\,\text{s}`", "**:math:`6.04\times` speedup**"
   "**Mean Consumer Stall**", ":math:`10.46\,\text{ms}`", ":math:`0.83\,\text{ms}`", ":math:`92.1\%` drop"
   "**P50 (Median) Stall**", ":math:`10.22\,\text{ms}`", ":math:`0.032\,\text{ms}` (:math:`32\,\mu\text{s}`)", "**:math:`99.7\%` drop**"
   "**P90 Stall**", ":math:`10.26\,\text{ms}`", ":math:`2.85\,\text{ms}`", ":math:`72.2\%` drop"
   "**P99 Stall**", ":math:`20.55\,\text{ms}`", ":math:`9.41\,\text{ms}`", ":math:`54.2\%` drop"

Because the 8 active streams execute their I/O concurrently in the background, the consumer
receives samples almost instantly (:math:`32\,\mu\text{s}` median wait), saving :math:`83.4\%`
of total wall-clock time.

Workload 2: Latency Jitter and Spikes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Simulating erratic disk/network access (:math:`90\%` normal :math:`8\,\text{ms}`, :math:`10\%`
spikes :math:`25\,\text{ms}`) with :math:`A=4` active streams and :math:`8\,\text{ms}` consumer work:

.. csv-table:: Jitter Absorption Results (250 items, :math:`A=4`)
   :header: "Metric", "Synchronous Baseline", "Prefetched (``buffer_size=4``)", "Reduction / Gain"
   :widths: 25, 25, 25, 25

   "**Total Wall-Clock Time**", ":math:`4.506\,\text{s}`", ":math:`2.073\,\text{s}`", "**:math:`2.17\times` speedup**"
   "**Mean Consumer Stall**", ":math:`9.91\,\text{ms}`", ":math:`0.17\,\text{ms}`", ":math:`98.3\%` drop"
   "**P50 (Median) Stall**", ":math:`8.26\,\text{ms}`", ":math:`0.060\,\text{ms}`", ":math:`99.3\%` drop"
   "**P95 Stall**", ":math:`24.82\,\text{ms}`", ":math:`0.121\,\text{ms}`", "**:math:`99.5\%` drop**"
   "**P99 Stall**", ":math:`27.59\,\text{ms}`", ":math:`0.166\,\text{ms}`", "**:math:`99.4\%` drop**"

The buffer completely shields the consumer from I/O spikes: 99% of samples are ready within
:math:`0.17\,\text{ms}` of request, compared to :math:`27.6\,\text{ms}` under synchronous execution.

Active Stream Scaling
^^^^^^^^^^^^^^^^^^^^^
Evaluating total runtime as the active set size :math:`A` increases (:math:`15\,\text{ms}` I/O, :math:`1\,\text{ms}` consumer):

.. csv-table:: Throughput Scaling with Active Set Size :math:`A`
   :header: ":math:`A`", "Synchronous Duration", "Prefetched Duration", "Speedup"
   :widths: 15, 30, 30, 25

   "**2**", ":math:`3.339\,\text{s}`", ":math:`1.667\,\text{s}`", "**:math:`2.00\times`**"
   "**4**", ":math:`3.324\,\text{s}`", ":math:`0.918\,\text{s}`", "**:math:`3.62\times`**"
   "**8**", ":math:`3.275\,\text{s}`", ":math:`0.546\,\text{s}`", "**:math:`6.00\times`**"
   "**16**", ":math:`3.265\,\text{s}`", ":math:`0.349\,\text{s}`", "**:math:`9.37\times`**"

Synchronous runtime is completely invariant to :math:`A` because only one stream runs at a time.
Prefetched runtime scales near-linearly with active streams, achieving a **:math:`9.37\times` speedup**
at :math:`A = 16`.

Buffer Depth Sensitivity
^^^^^^^^^^^^^^^^^^^^^^^^
Evaluating buffer capacity :math:`B \in \{1, 2, 4, 8\}` with :math:`A = 4`, :math:`10\,\text{ms}` I/O, :math:`6\,\text{ms}` consumer:

.. csv-table:: Buffer Depth Sensitivity
   :header: "Configuration", "Total Wall Time", "Mean Consumer Wait", "P50 Consumer Wait"
   :widths: 25, 25, 25, 25

   "**Sync (``None``)**", ":math:`3.318\,\text{s}`", ":math:`10.45\,\text{ms}`", ":math:`10.22\,\text{ms}`"
   "**``buffer_size=1``**", ":math:`1.268\,\text{s}`", ":math:`0.21\,\text{ms}`", ":math:`0.057\,\text{ms}`"
   "**``buffer_size=2``**", ":math:`1.251\,\text{s}`", ":math:`0.13\,\text{ms}`", ":math:`0.056\,\text{ms}`"
   "**``buffer_size=4``**", ":math:`1.251\,\text{s}`", ":math:`0.13\,\text{ms}`", ":math:`0.056\,\text{ms}`"
   "**``buffer_size=8``**", ":math:`1.254\,\text{s}`", ":math:`0.14\,\text{ms}`", ":math:`0.058\,\text{ms}`"

Even a modest buffer depth of **:math:`B = 2`** unlocks :math:`> 98\%` of potential gains,
meaning memory footprint can remain minimal while completely eliminating pipeline stalls.


Practical Usage Guide
---------------------

Enabling Prefetching in Mux Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Prefetching is controlled via the ``buffer_size`` keyword argument across all Pescador multiplexers:
:class:`~pescador.StochasticMux`, :class:`~pescador.ShuffledMux`, :class:`~pescador.RoundRobinMux`,
and :class:`~pescador.ChainMux`.

By default, ``buffer_size=None`` preserves 100% backward-compatible synchronous operation.
Setting ``buffer_size`` to a positive integer (recommended: :math:`2` to :math:`4`) activates concurrent
background prefetching.

.. code-block:: python

   import pescador

   # Define data generator
   def load_audio_features(file_path):
       for chunk in read_file(file_path):
           yield {'features': chunk}

   # Create streamers
   streamers = [
       pescador.Streamer(load_audio_features, path)
       for path in audio_files
   ]

   # Create a StochasticMux with concurrent background prefetching
   mux = pescador.StochasticMux(
       streamers,
       n_active=8,
       rate=64,
       buffer_size=4,       # Enable prefetching: 4 items per active stream
   )

   # Always consume within a context manager for deterministic resource cleanup
   with mux:
       for batch in mux.iterate(max_iter=10000):
           train_step(batch)


The Importance of the Context Manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When ``buffer_size > 0``, the multiplexer manages background threads and async event loops.
Always use the ``with mux:`` context manager pattern.
The context manager ensures that:

1. Background worker tasks are initiated when entering the block.
2. If an exception occurs in the consumer loop (e.g. keyboard interrupt, out-of-memory error),
   all background tasks and threads are promptly terminated without hanging or resource leaks.

.. code-block:: python

   # Recommended pattern:
   with mux:
       for item in mux:
           process(item)

   # Alternatively, mux.iterate() can manage the context automatically:
   for item in mux.iterate(max_iter=500):
       process(item)


Standalone Stream Prefetching: ``PrefetchStreamer``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have a single generator or pipeline that does not require multiplexing, you can wrap it
directly with :class:`~pescador.PrefetchStreamer`:

.. code-block:: python

   # Standalone data stream
   base_streamer = pescador.Streamer(data_generator, dataset_path)

   # Wrap with concurrent prefetching
   prefetched_streamer = pescador.PrefetchStreamer(
       base_streamer,
       buffer_size=4,
   )

   with prefetched_streamer:
       for sample in prefetched_streamer:
           model.train_on_batch(sample)


Choosing the Optimal Buffer Depth
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The optimal value of ``buffer_size`` depends on sample size and memory constraints:

* **Small items (metadata, audio frame vectors, tabular rows)**:
  Use ``buffer_size = 4`` to ``8``. Memory overhead is negligible, and deeper buffers provide
  maximum protection against I/O jitter.
* **Large items (high-resolution images, large multidimensional arrays, raw audio)**:
  Use ``buffer_size = 1`` or ``2``. As established in the sensitivity analysis above,
  ``buffer_size = 2`` provides over :math:`95\%` of the maximum latency reduction while strictly
  bounding RAM usage.
* **CPU-Bound vs. I/O-Bound**:
  Prefetching uses Python threads within an asynchronous event loop, which excels at I/O-bound tasks
  (disk reads, network calls, C-extension processing releasing the GIL such as NumPy/SoundFile).
  For heavily CPU-bound Python generator logic that holds the GIL, consider using
  :ref:`ZMQStreamer` to distribute generation across separate operating system processes.
