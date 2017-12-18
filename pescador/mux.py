'''Stream multiplexing

Defines the interface and several varieties of "Mux". A "Mux" is
a Streamer which wraps N other streamers, and at every step yields a
sample from one of its sub-streamers.

This module defines the following Mux types:

`PoissonMux`

    A Mux which chooses its active streams stochastically, and chooses
    samples from the active streams stochastically. `PoissonMux` is equivalent
    to the `pescador.Mux` from versions <2.0.

     `PoissonMux` has a `mode` parameter which selects how it operates, with
     the following modes:

    `with_replacement`

        Sample streamers with replacement.  This allows a single stream to
        be used multiple times (even simultaneously).

    `exhaustive`

        Each streamer is consumed at most once and never
        revisited.

    `single_active`

        Each stream in the candidate pool is either active or not.
        Streams are revived when they are exhausted.
        This setting makes it so that streams in the
        active pool are *uniquely* selected from the candidate pool, where as
        `with_replacement` allows the same stream to be used more than once.

`ShuffledMux`

    A `ShuffledMux` samples from each provided streamer with equal probability.

`RoundRobinMux`

    Iterates over all the streamers in strict order.

`ChainMux`

    As in itertools.chain(), runs the first streamer to exhaustion, then
    the second, then the third, etc. Uses only a single stream at a time.

`Mux`

    The pescador<2.0 `Mux` is still available and works the same,
    but is deprecated.
'''
from warnings import warn
import six
import numpy as np

from . import core
from .exceptions import PescadorError


class Mux(core.Streamer):
    '''Stochastic multiplexor for Streamers

    Examples
    --------
    >>> # Create a collection of streamers
    >>> seeds = [pescador.Streamer(my_generator) for i in range(10)]
    >>> # Multiplex them together into a single streamer
    >>> # Use at most 3 streams at once
    >>> mux = pescador.Mux(seeds, k=3)
    >>> for batch in mux():
    ...     MY_FUNCTION(batch)

    Mux([stream, range(8), stream2])
    '''
    def __init__(self, streamers, k,
                 rate=256.0, weights=None, with_replacement=True,
                 prune_empty_streams=True, revive=False,
                 random_state=None):
        """Given an array (pool) of streamer types, do the following:

        1. Select ``k`` streams at random to iterate from
        2. Assign each activated stream a sample count ~ Poisson(lam)
        3. Yield samples from the streams by randomly multiplexing
           from the active set.
        4. When a stream is exhausted, select a new one from `streamers`.

        Parameters
        ----------
        streamers : iterable of streamers
            The collection of streamer-type objects

        k : int > 0
            The number of streams to keep active at any time.

        rate : float > 0 or None
            Rate parameter for the Poisson distribution governing sample counts
            for individual streams.
            If ``None``, sample infinitely from each stream.

        weights : np.ndarray or None
            Optional weighting for ``streamers``.
            If ``None``, then weights are assumed to be uniform.
            Otherwise, ``weights[i]`` defines the sampling proportion
            of ``streamers[i]``.

            Must have the same length as ``streamers``.

        with_replacement : bool
            Sample streamers with replacement.  This allows a single stream to
            be used multiple times (even simultaneously).
            If ``False``, then each streamer is consumed at most once and never
            revisited.

        prune_empty_streams : bool
            Disable streamers that produce no data.
            If ``True``, streamers that previously produced no data are never
            revisited.
            Note:
            1. This may be undesireable for streams where past emptiness
            may not imply future emptiness.
            2. Failure to prune truly empty streams with `revive=True` can
            result in infinite looping behavior. Disable with caution.

        revive: bool
            If ``with_replacement`` is ``False``, setting ``revive=True``
            will re-insert previously exhausted streams into the candidate set.

            This configuration allows a stream to be active at most once at any
            time.

        random_state : None, int, or np.random.RandomState
            If int, random_state is the seed used by the random number
            generator;

            If RandomState instance, random_state is the random number
            generator;

            If None, the random number generator is the RandomState instance
            used by np.random.
        """
        warn('`Mux` is deprecated in pescador 2.0. '
             'Please use `PoissonMux` instead to maintain forward-compatibility.'
             'This class will be removed in pescador 2.1.', DeprecationWarning)

        self.streamers = streamers
        self.n_streams = len(streamers)
        self.k = k
        self.rate = rate
        self.weights = weights
        self.with_replacement = with_replacement
        self.prune_empty_streams = prune_empty_streams
        self.revive = revive

        self.deactivate()

        if random_state is None:
            self.rng = np.random
        elif isinstance(random_state, int):
            self.rng = np.random.RandomState(seed=random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.rng = random_state
        else:
            raise PescadorError('Invalid random_state={}'.format(random_state))

        if not self.n_streams:
            raise PescadorError('Cannot mux an empty collection')

        if self.weights is None:
            self.weights = 1. / self.n_streams * np.ones(self.n_streams)

        self.weights = np.atleast_1d(self.weights)

        if len(self.weights) != len(self.streamers):
            raise PescadorError('`weights` must be the same '
                                'length as `streamers`')

        if not (self.weights > 0.0).any():
            raise PescadorError('`weights` must contain at least '
                                'one positive value')

        self.weights /= np.sum(self.weights)

    def activate(self):
        """Activates a number of streams"""
        self.distribution_ = 1. / self.n_streams * np.ones(self.n_streams)
        self.valid_streams_ = np.ones(self.n_streams, dtype=bool)

        self.streams_ = [None] * self.k

        self.stream_weights_ = np.zeros(self.k)
        self.stream_counts_ = np.zeros(self.k, dtype=int)
        # Array of pointers into `self.streamers`
        self.stream_idxs_ = np.zeros(self.k, dtype=int)

        for idx in range(self.k):

            if not (self.distribution_ > 0).any():
                break

            self.stream_idxs_[idx] = self.rng.choice(
                self.n_streams, p=self.distribution_)
            self.streams_[idx], self.stream_weights_[idx] = (
                self._new_stream(self.stream_idxs_[idx]))

        self.weight_norm_ = np.sum(self.stream_weights_)

    def deactivate(self):
        self.streams_ = None
        self.stream_weights_ = None
        self.stream_counts_ = None
        self.stream_idxs_ = None
        self.weight_norm_ = None

    def iterate(self, max_iter=None):
        # Calls Streamer's __enter__, which calls activate()
        with self:

            # Main sampling loop
            n = 0

            if max_iter is None:
                max_iter = np.inf

            while n < max_iter and self.weight_norm_ > 0.0:
                # Pick a stream from the active set
                idx = self.rng.choice(self.k, p=(self.stream_weights_ /
                                                 self.weight_norm_))

                # Can we sample from it?
                try:
                    # Then yield the sample
                    yield six.advance_iterator(self.streams_[idx])

                    # Increment the sample counter
                    n += 1
                    self.stream_counts_[idx] += 1

                except StopIteration:
                    # Oops, this one's exhausted.

                    if (self.prune_empty_streams and
                            self.stream_counts_[idx] == 0):
                        # If we're disabling empty seeds, see if this stream
                        # produced data; if it didn't, turn it off.
                        self.distribution_[self.stream_idxs_[idx]] = 0.0
                        self.valid_streams_[self.stream_idxs_[idx]] = False

                    if self.revive and not self.with_replacement:
                        # When this case is hit, the `distribution_` for
                        # this "seed"/"stream" is 0.0, because it got set
                        # to when we activated it. (in `_new_stream`)

                        # Since revive mode is on, we set it to the max
                        # current probability to enable it to be used again.
                        if self.distribution_.any():
                            self.distribution_[self.stream_idxs_[idx]] = (
                                np.max(self.distribution_))
                        else:
                            self.distribution_[self.stream_idxs_[idx]] = 1.0

                    if (self.distribution_ > 0).any():
                        # Replace it and move on if there are still seeds
                        # in the pool.
                        self.distribution_[:] /= np.sum(self.distribution_)

                        self.stream_idxs_[idx] = self.rng.choice(
                            self.n_streams, p=self.distribution_)

                        self.streams_[idx], self.stream_weights_[idx] = (
                            self._new_stream(self.stream_idxs_[idx]))

                        self.stream_counts_[idx] = 0

                    else:
                        # Otherwise, this one's exhausted.
                        # Set its probability to 0
                        self.stream_weights_[idx] = 0.0

                    self.weight_norm_ = np.sum(self.stream_weights_)

                # If everything has been pruned, kill the while loop
                if not self.valid_streams_.any():
                    break

    def _new_stream(self, idx):
        '''Randomly select and create a stream.

        Parameters
        ----------
        idx : int, [0:n_streams - 1]
            The stream index to replace
        '''
        if len(self.streamers) != len(self.weights):
            raise PescadorError('`streamers` must have the same '
                                'length as `weights`')

        if len(self.streamers) != len(self.distribution_):
            raise PescadorError('`streamers` must have the same '
                                'length as `distribution`')

        # instantiate
        if self.rate is not None:
            n_stream = 1 + self.rng.poisson(lam=self.rate)
        else:
            n_stream = None

        # If we're sampling without replacement, zero this one out
        # This effectively disables this stream as soon as it is chosen,
        # preventing it from being chosen again (unless it is revived)
        if not self.with_replacement:
            self.distribution_[idx] = 0.0

            # Correct the distribution
            if (self.distribution_ > 0).any():
                self.distribution_[:] /= np.sum(self.distribution_)

        return (self.streamers[idx].iterate(max_iter=n_stream),
                self.weights[idx])


class BaseMux(core.Streamer):
    """BaseMux defines the interface to a Mux. Fundamentally, a Mux
    is a container for multiple Streamers, which selects a Sample from one of
    its streamers at every iteration.

    A Mux has the following fundamental behaviors:

     * When "activated", choose a subset of available streamers to stream from
       (the "active substreams")
     * When a sample is drawn from the mux (via iterate),
       chooses which active substream to stream from.
     * Handles exhaustion of streams (restarting, replacing, ...)
    """
    def __init__(self, streamers, random_state=None):
        """
        Parameters
        ----------
        streamers : iterable of streamers
            The collection of streamer-type objects

        random_state : None, int, or np.random.RandomState
            If int, random_state is the seed used by the random number
            generator;

            If RandomState instance, random_state is the random number
            generator;

            If None, the random number generator is the RandomState instance
            used by np.random.
        """
        self.streamers = streamers

        if random_state is None:
            self.rng = np.random
        elif isinstance(random_state, int):
            self.rng = np.random.RandomState(seed=random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.rng = random_state
        else:
            raise PescadorError('Invalid random_state={}'.format(random_state))

        # Clear state and reset actiave/deactivate params.
        self.deactivate()

    @property
    def n_streams(self):
        """Return the number of streamers.

        Note: Some `Mux` sub-classes (i.e. `ChainMux`) may allow generators
        of streamers, in which case this will fail.
        """
        return len(self.streamers)

    def activate(self):
        """Activates the mux as a streamer, choosing which substreams to
        select as active.

        Any implementation of `activate()` should implement the following:

        1. Create a pool of active streams, `self.streams_`.
            This is required by the `BaseMux.iterate()`, which is the core
             of all Mux.

        2. Must call `self._new_stream(idx)` to setup each active stream
            for the `Mux`.

        3. Must setup `self.stream_counts_`; an array of zeros of the same
            length as `self.streams_` which counts how many samples have been
            drawn from each active streamer. This is used to determine if
            the streamer has produced any samples.
        """
        raise NotImplementedError()

    def deactivate(self):
        """Reset the Mux state."""
        pass

    def iterate(self, max_iter=None):
        """Yields items from the mux, and handles stream exhaustion and
        replacement.
        """
        if max_iter is None:
            max_iter = np.inf

        # Calls Streamer's __enter__, which calls activate()
        with self:
            # Main sampling loop
            n = 0

            while n < max_iter and self._streamers_available():
                # Pick a stream from the active set
                idx = self._next_sample_index()

                # Can we sample from it?
                try:
                    # Then yield the sample
                    yield six.advance_iterator(self.streams_[idx])

                    # Increment the sample counter
                    n += 1
                    self.stream_counts_[idx] += 1

                except StopIteration:
                    # Oops, this stream is exhausted.

                    # Call child-class exhausted-stream behavior
                    self._on_stream_exhausted(idx)

                    # Setup a new stream for this index
                    self._replace_stream(idx)

    def _streamers_available(self):
        "Override this to modify the behavior of the main iter loop condition."
        return True

    def _on_stream_exhausted(self, idx):
        """Override this to provide a Mux with additional behavior
        when a stream is exhausted. This gets called *after* streams
        are pruned, but before replacing it with a new stream. Handle
        behaviors for closing down your previous stream here.

        Parameters
        ----------
        idx : int, [0:k - 1]
            Index of the exhausted stream.
        """
        pass

    def _replace_stream(self, idx):
        """Called after a stream has been exhausted, replace the stream
        with another from the pool.

        Any implementation of `_replace_stream()` should call

        > # Setup the new stream.
        > self._new_stream(idx)

        to replace the exhausted stream.
        """
        raise NotImplementedError("_replace_stream() must be implemented in"
                                  " a child class.")

    def _new_stream(self, idx):
        """Activate a new stream, given the index into the stream pool.

        Parameters
        ----------
        idx : int, [0:n_streams - 1]
            The stream index to replace
        """
        raise NotImplementedError("_new_stream() must be implemented in"
                                  " a child class.")

    def _next_sample_index(self):
        """Returns the index in self.streams_ for the streamer from which
        to draw the next sample.

        Implementation required in any child class.
        """
        raise NotImplementedError("_next_sample_index() must be implemented in"
                                  " a child class.")


class PoissonMux(BaseMux):
    '''Stochastic Mux

    Examples
    --------
    >>> # Create a collection of streamers
    >>> seeds = [pescador.Streamer(my_generator) for i in range(10)]
    >>> # Multiplex them together into a single streamer
    >>> # Use at most 3 streams at once
    >>> mux = pescador.PoissonMux(seeds, k=3)
    >>> for batch in mux():
    ...     MY_FUNCTION(batch)

    PoissonMux([stream, range(8), stream2])
    '''
    def __init__(self, streamers, k_active, rate,
                 weights=None,
                 mode="with_replacement",
                 prune_empty_streams=True,
                 random_state=None):
        """Given an array (pool) of streamer types, do the following:

        1. Select ``k`` streams at random to iterate from
        2. Assign each activated stream a sample count ~ Poisson(rate)
        3. Yield samples from the streams by randomly multiplexing
           from the active set.
        4. When a stream is exhausted, select a new one from `streamers`.

        Parameters
        ----------
        streamers : iterable of streamers
            The collection of streamer-type objects

        k_active : int > 0
            The number of streams to keep active at any time.

        rate : float > 0 or None
            Rate parameter for the Poisson distribution governing sample counts
            for individual streams.
            If ``None``, sample infinitely from each stream.

        weights : np.ndarray or None
            Optional weighting for ``streamers``.
            If ``None``, then weights are assumed to be uniform.
            Otherwise, ``weights[i]`` defines the sampling proportion
            of ``streamers[i]``.

            Must have the same length as ``streamers``.

        mode : ["with_replacement", "single_active", "exhaustive"]
            with_replacement
                Sample streamers with replacement.  This allows a single
                stream to be used multiple times (even simultaneously).
                Streams are sampled independently and indefinitely.

            single_active
                This configuration allows a stream to be active at most once
                at any time.

            exhaustive
                Each streamer is consumed at most once and never revisited.
                Run every selected stream once to exhaustion.

        prune_empty_streams : bool
            Disable streamers that produce no data. See `BaseMux`

        random_state : None, int, or np.random.RandomState
            See `BaseMux`
        """
        self.mode = mode
        self.k_active = k_active
        self.rate = rate
        self.prune_empty_streams = prune_empty_streams

        super(PoissonMux, self).__init__(
            streamers, random_state=random_state)

        if not self.n_streams:
            raise PescadorError('Cannot mux an empty collection')

        if self.mode not in [
                "with_replacement", "single_active", "exhaustive"]:
            raise PescadorError("{} is not a valid mode for PoissonMux".format(
                self.mode))

        self.weights = weights
        if self.weights is None:
            self.weights = 1. / self.n_streams * np.ones(self.n_streams)
        self.weights = np.atleast_1d(self.weights)

        if len(self.weights) != len(self.streamers):
            raise PescadorError('`weights` must be the same '
                                'length as `streamers`')

        if not (self.weights > 0.0).any():
            raise PescadorError('`weights` must contain at least '
                                'one positive value')

        self.weights /= np.sum(self.weights)

    def activate(self):
        # These do not depend on the number of streams, k
        self.distribution_ = 1. / self.n_streams * np.ones(self.n_streams)
        self.valid_streams_ = np.ones(self.n_streams, dtype=bool)

        if len(self.streamers) != len(self.distribution_):
            raise PescadorError('`streamers` must have the same '
                                'length as `distribution`')

        # But the following do depend on the number of active streams.
        # The active streamers
        self.streams_ = [None] * self.k_active

        # Weights of the active streams.
        # Once a stream is exhausted, it is set to 0
        self.stream_weights_ = np.zeros(self.k_active)
        # How many samples have been draw from each (active) stream.
        self.stream_counts_ = np.zeros(self.k_active, dtype=int)
        # Array of pointers into `self.streamers`
        self.stream_idxs_ = np.zeros(self.k_active, dtype=int)

        # Initialize each active stream.
        for idx in range(self.k_active):

            if not (self.distribution_ > 0).any():
                break

            # Setup a new streamer at this index.
            self._new_stream(idx)

        self.weight_norm_ = np.sum(self.stream_weights_)

    def deactivate(self):
        self.distribution_ = np.zeros(self.n_streams)
        self.valid_streams_ = np.zeros(self.n_streams)

        self.streams_ = None
        self.stream_idxs_ = None
        self.stream_counts_ = None
        self.stream_weights_ = None
        self.weight_norm_ = None

    def _streamers_available(self):
        return self.weight_norm_ > 0.0 and self.valid_streams_.any()

    def _next_sample_index(self):
        """PoissonMux chooses its next sample stream randomly"""
        return self.rng.choice(self.k_active,
                               p=(self.stream_weights_ /
                                  self.weight_norm_))

    def _on_stream_exhausted(self, idx):
        # If we're disabling empty seeds, see if this stream
        # produced any data; if it didn't, turn it off.
        if (self.prune_empty_streams is True and
                self.stream_counts_[idx] == 0):
            self.distribution_[self.stream_idxs_[idx]] = 0.0
            self.valid_streams_[self.stream_idxs_[idx]] = False

        # This is the same as
        #  if self.revive and not self.with_replacement in the original Mux
        if self.mode == "single_active":
            # If we need to revive a seed, give it the max
            # current probability
            if self.distribution_.any():
                self.distribution_[self.stream_idxs_[idx]] = (
                    np.max(self.distribution_))
            else:
                self.distribution_[self.stream_idxs_[idx]] = 1.0

    def _activate_stream(self, idx):
        '''Randomly select and create a stream.

        PoissonMux adds mode handling to _activate_stream, making it so that
        if we're not sampling "with_replacement", the distribution for this
        chosen streamer is set to 0, causing the streamer not to be available
        until it is exhausted.

        Parameters
        ----------
        idx : int, [0:n_streams - 1]
            The stream index to replace
        '''
        # Get the number of samples for this streamer.
        n_samples_to_stream = None
        if self.rate is not None:
            n_samples_to_stream = 1 + self.rng.poisson(lam=self.rate)

        # instantiate a new streamer
        streamer = self.streamers[idx].iterate(max_iter=n_samples_to_stream)
        weight = self.weights[idx]

        # If we're sampling without replacement, zero this one out
        # This effectively disables this stream as soon as it is chosen,
        # preventing it from being chosen again (unless it is revived)
        # if not self.with_replacement:
        if self.mode != "with_replacement":
            self.distribution_[idx] = 0.0

            # Correct the distribution
            if (self.distribution_ > 0).any():
                self.distribution_[:] /= np.sum(self.distribution_)

        return streamer, weight

    def _new_stream(self, idx):
        '''Randomly select and create a new stream.

        Parameters
        ----------
        idx : int, [0:n_streams - 1]
            The stream index to replace
        '''
        # Choose the stream index from the candidate pool
        self.stream_idxs_[idx] = self.rng.choice(
            self.n_streams, p=self.distribution_)

        # Activate the Streamer, and get the weights
        self.streams_[idx], self.stream_weights_[idx] = self._activate_stream(
            self.stream_idxs_[idx])

        # Reset the sample count to zero
        self.stream_counts_[idx] = 0

    def _replace_stream(self, idx):
        # If there are active streams reamining,
        # choose a new one to make active.
        if (self.distribution_ > 0).any():
            # Replace it and move on if there are still seeds
            # in the pool.
            self.distribution_[:] /= np.sum(self.distribution_)

            # Setup a new streamer at this index.
            self._new_stream(idx)
        else:
            # Otherwise, this one's exhausted.
            # Set its probability to 0
            self.stream_weights_[idx] = 0.0

        self.weight_norm_ = np.sum(self.stream_weights_)


class ShuffledMux(BaseMux):
    """A variation on a mux, which takes N streamers, and samples
    from them equally, guaranteeing all N streamers to be "active",
    unlike the base Mux, which randomly chooses streams when activating.

    `ShuffledMux` automatically restarts streams when they die. For a more
    nuanced behavior, consider using `PoissonMux` with `single_active=True`.
    """
    def __init__(self, streamers, weights=None,
                 random_state=None):
        """
        Parameters
        ----------
        streamers : iterable of streamers
            The collection of streamer-type objects

        weights : np.ndarray or None
            Optional weighting for ``streamers``.
            If ``None``, then weights are assumed to be uniform.
            Otherwise, ``weights[i]`` defines the sampling proportion
            of ``streamers[i]``.

            Must have the same length as ``streamers``.

        random_state : None, int, or np.random.RandomState
            If int, random_state is the seed used by the random number
            generator;

            If RandomState instance, random_state is the random number
            generator;

            If None, the random number generator is the RandomState instance
            used by np.random.
        """
        super(ShuffledMux, self).__init__(
            streamers,
            random_state=random_state)

        if not self.n_streams:
            raise PescadorError('Cannot mux an empty collection')

        self.weights = weights
        if self.weights is None:
            self.weights = 1. / self.n_streams * np.ones(self.n_streams)
        self.weights = np.atleast_1d(self.weights)

        if len(self.weights) != len(self.streamers):
            raise PescadorError('`weights` must be the same '
                                'length as `streamers`')

        if not (self.weights > 0.0).any():
            raise PescadorError('`weights` must contain at least '
                                'one positive value')

        self.weights /= np.sum(self.weights)

    def activate(self):
        """ShuffledMux's activate is similar to PoissonMux,
        but there is no 'k_active', since all the streams are always available.
        """
        self.streams_ = [None] * self.n_streams

        # Weights of the active streams.
        # Once a stream is exhausted, it is set to 0.
        # Upon activation, this is just a copy of self.weights.
        self.stream_weights_ = np.array(self.weights, dtype=float)
        # How many samples have been drawn from each (active) stream.
        self.stream_counts_ = np.zeros(self.n_streams, dtype=int)

        # Initialize each active stream.
        for idx in range(self.n_streams):
            # Setup a new streamer at this index.
            self._new_stream(idx)

        self.weight_norm_ = np.sum(self.stream_weights_)

    def deactivate(self):
        self.streams_ = None
        self.stream_weights_ = None
        self.stream_counts_ = None
        self.weight_norm_ = None

    def _streamers_available(self):
        return self.weight_norm_ > 0.0

    def _next_sample_index(self):
        """ShuffledMux chooses its next sample stream randomly,
        conditioned on the stream weights.
        """
        return self.rng.choice(self.n_streams,
                               p=(self.stream_weights_ /
                                  self.weight_norm_))

    def _on_stream_exhausted(self, idx):
        # See if this stream produced any data; if it didn't, turn it off
        # using the stream weights.
        # stream_weights_ only get modified if the stream produced no data.
        if self.stream_counts_[idx] == 0:
            self.stream_weights_[idx] = 0

    def _new_stream(self, idx):
        '''Randomly select and create a new stream.

        Parameters
        ----------
        idx : int, [0:n_streams - 1]
            The stream index to replace
        '''
        # Don't activate the stream if the weight is 0 or None
        if self.stream_weights_[idx]:
            self.streams_[idx] = self.streamers[idx].iterate()
        else:
            self.streams_[idx] = None

        # Reset the sample count to zero
        self.stream_counts_[idx] = 0

    def _replace_stream(self, idx):
        # If there are active streams reamining,
        # choose a new one to make active.
        if (self.stream_weights_ > 0).any():
            # Setup a new streamer at this index.
            self._new_stream(idx)
        else:
            # Otherwise, this one's exhausted.
            # Set its probability to 0
            # In practice, this is probably unnecessary.
            self.stream_weights_[idx] = 0.0

        self.weight_norm_ = np.sum(self.stream_weights_)


class RoundRobinMux(BaseMux):
    """A Mux which iterates over all streamers in strict order.

    Based on the roundrobin() example in python itertools:
     https://docs.python.org/3/library/itertools.html#itertools-recipes

    TODO: (maybe) handle stream exhaustion?

    Examples
    --------
    >>> a = pescador.Streamer("a")
    >>> b = pescador.Streamer("b")
    >>> c = pescador.Streamer("c")
    >>> mux = pescador.RoundRobinMux([a, b, c])
    >>> print("".join(mux.iterate(9)))
    "abc"

    >>> mux = pescador.RoundRobinMux([a, b, c], mode="cycle")
    >>> print("".join(mux.iterate(9)))
    "abcabcabc"

    >>> mux = pescador.RoundRobinMux([a, b, c], mode="permuted_cycle")
    >>> print("".join(mux.iterate(20)))
    "abcacbacbacbbcabacac"
    """
    def __init__(self, streamers, mode="exhaustive", random_state=None):
        """
        Parameters
        ----------
        streamers : list of pescador.Streamers

        mode : ["exhaustive", "cycle", "permuted_cycle"]
            `exhaustive`
                `RoundRobinMux` will exit after each stream has been exhausted.

            `cycle`
                Restart streamer once all streams are exhausted.

            `permuted_cycle`
                Restart streamer once streams are exhausted, and permute
                the order of the streams.

        random_state : None, int, or np.random.RandomState
            If int, random_state is the seed used by the random number
            generator;

            If RandomState instance, random_state is the random number
            generator;

            If None, the random number generator is the RandomState instance
            used by np.random.
        """
        self.mode = mode
        super(RoundRobinMux, self).__init__(
            streamers,
            random_state=random_state)

        if not self.n_streams:
            raise PescadorError('Cannot mux an empty collection')

    def activate(self):
        self._setup_streams(False)

    def deactivate(self):
        self.active_index_ = None
        self.streams_ = None
        self.stream_idxs_ = None
        self.stream_counts_ = None

    def _setup_streams(self, permute=False):
        self.active_index_ = 0

        # The active streamers
        self.streams_ = [None] * self.n_streams

        # Stream pointers; this is used to optionally shuffle the order
        # for each complete iteration.
        self.stream_idxs_ = np.arange(self.n_streams, dtype=int)

        if permute:
            self.rng.shuffle(self.stream_idxs_)

        # How many samples have been drawn from each?
        self.stream_counts_ = np.zeros(self.n_streams, dtype=int)

        # Initialize each active stream.
        for idx in range(self.n_streams):
            # Setup a new streamer at this index.
            self._new_stream(idx)

    def _streamers_available(self):
        """Check if any of streams_ is not a None; if they're all None,
        the streamers have all been exhausted.
        """
        return any([x is not None for x in self.streams_])

    def _next_sample_index(self):
        """Rotates through each active sampler by incrementing the index"""
        # Return the next streamer index where the streamer is not None,
        # wrapping around.
        idx = self.active_index_
        self.active_index_ += 1

        if self.active_index_ >= len(self.streams_):
            self.active_index_ = 0

        # Continue to increment if this streamer is exhausted (None)
        # This should never be infinite looping;
        # the `_streamers_available` check happens immediately
        # before this, so there should always be at least one not-None
        # streamer.
        while self.streams_[idx] is None:
            idx = self.active_index_
            self.active_index_ += 1

            if self.active_index_ >= len(self.streams_):
                self.active_index_ = 0

        return idx

    def _new_stream(self, idx):
        """Activate a new stream, given the index into the stream pool.

        BaseMux's _new_stream simply chooses a new stream and activates it.
        For special behavior (ie Weighted streams), you must override this
        in a child class.

        Parameters
        ----------
        idx : int, [0:n_streams - 1]
            The stream index to replace
        """
        # Get the stream index from the candidate pool
        stream_index = self.stream_idxs_[idx]

        # Activate the Streamer, and get the weights
        self.streams_[idx] = self.streamers[stream_index].iterate()

        # Reset the sample count to zero
        self.stream_counts_[idx] = 0

    def _replace_stream(self, idx=None):
        """Called by `BaseMux`'s iterate() when a stream is exhausted.
        Set the stream to None so it is ignored once exhausted.

        Parameters
        ----------
        idx : int or None

        Raises
        ------
        StopIteration
            If all streams are consumed, and `mode`=="exahustive"
        """
        self.streams_[idx] = None

        # Check if we've now exhausted all the streams.
        if not self._streamers_available():
            if self.mode == 'exhaustive':
                pass

            elif self.mode == "cycle":
                self._setup_streams(permute=False)

            elif self.mode == "permuted_cycle":
                self._setup_streams(permute=True)


class ChainMux(BaseMux):
    """As in itertools.chain(). Runs the first streamer to exhaustion,
    then the second, then the third, etc.

    Examples
    --------
    # Run Chain once through until the end.
    >>> a = pescador.Streamer("abc")
    >>> b = pescador.Streamer("def")
    >>> mux = pescador.mux.ChainMux([a, b], mode="exhaustive")
    >>> "".join(list(mux.iterate()))
    "abcdef"

    # Chain restarts from the beginning once exhausted.
    >>> a = pescador.Streamer("abc")
    >>> b = pescador.Streamer("def")
    >>> mux = pescador.mux.ChainMux([a, b], mode="cycle")
    >>> "".join(list(mux.iterate(max_iter=12)))
    "abcdefabcdef"

    # Chain a generator of streamers
    >>> import string
    def gen_streamers(n_streamers, n_copies):
        for i in range(n_streamers):
            yield pescador.Streamer(string.ascii_letters[i] * n_copies)

    >>> mux = pescador.mux.ChainMux(gen_streamers(3, 5))
    >>> "".join(list(mux.iterate()))
    "aaaaabbbbbccccc"
    """
    def __init__(self, streamers, mode="exhaustive",
                 random_state=None):
        """
        Parameters
        ----------
        streamers : list of pescador.Streamers OR generator of
            pescador.Streamrers

        mode : ["exhaustive", "cycle"]
            `exhaustive`
                `ChainMux will exit after each stream has been exhausted.

            `cycle`
                `ChainMux will restart from the beginning after each
                streamer has been run to exhaustion.

        random_state : None, int, or np.random.RandomState
            If int, random_state is the seed used by the random number
            generator;

            If RandomState instance, random_state is the random number
            generator;

            If None, the random number generator is the RandomState instance
            used by np.random.
        """
        super(ChainMux, self).__init__(
            streamers, random_state=random_state)

        self.mode = mode

    def activate(self):
        # Use a streamer to iterate over the input streamers.
        # This allows the streamers to be an iterable, and also easily
        #  be restarted.
        self.chain_streamer_ = core.Streamer(self.streamers)

        # Activate the chain_streamer_, initializing the generator, and
        # getting the first stream.
        self.stream_generator_ = self.chain_streamer_.iterate()

        # Chainmux only ever has one active streamer.
        self.streams_ = [None]
        self.stream_counts_ = np.zeros(1, dtype=int)

        # Initialize the active stream.
        # Setup a new streamer at this index.
        self._new_stream()

    def deactivate(self):
        self.chain_streamer_ = None
        self.chain_generator_ = None
        self.streams_ = None
        self.stream_counts_ = None

    def _streamers_available(self):
        """As we are treating `streamers` as a generator, we can only know
        if it is available if the streamer exited or not.
        """
        return self.chain_streamer_.active

    def _next_sample_index(self):
        """There is only one streamer to choose from; always 0"""
        return 0

    def _replace_stream(self, idx=None):
        """Called by `BaseMux`'s iterate() when a stream is exhausted.

        Parameters
        ----------
        idx : int or None
            Passed from the `BaseMux` to indicate which streamer index
            was exhausted. For `ChainMux`, there is only one active streamer,
            so it is just ignored.
        """
        self._new_stream()

    def _new_stream(self):
        '''Grab the next stream from the input streamers, and start it.

        Raises
        ------
        StopIteration
            When the input list or generator of streamers is complete,
            will raise a StopIteration. If `mode == cycle`, it
            will instead restart iterating from the beginning of the sequence.
        '''
        try:
            # Advance the stream_generator_ to get the next available stream.
            # If successful, this will make self.chain_streamer_.active True
            next_stream = six.advance_iterator(self.stream_generator_)

        except StopIteration:
            # If running with cycle, restart the chain_streamer_ after
            # exhaustion.
            if self.mode == "cycle":
                self.stream_generator_ = self.chain_streamer_.iterate()

                # Try again to get the next stream;
                # if it fails this time, just let it raise the StopIteration;
                # this means the streams are probably dead or empty.
                next_stream = six.advance_iterator(self.stream_generator_)

            # If running in exhaustive mode
            else:
                # self.chain_streamer_ should no longer be active, so
                # the outer loop should fall out without running.
                next_stream = None

        if next_stream is not None:
            # Start that stream, and return it.
            streamer = next_stream.iterate()

            # Activate the Streamer
            self.streams_[0] = streamer

            # Reset the sample count to zero
            self.stream_counts_[0] = 0
