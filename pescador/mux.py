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

    A `ShuffledMux` uses all the given streamers, and samples from
    each of them with equal probability.

`RoundRobinMux`

    As in `ShuffledMux`, uses all the given streamers, but iterates over
    the streamers in strict order.

`ChainMux`

    As in itertools.chain(), runs the first streamer to exhaustion, then
    the second, then the third, etc. Uses only a single stream at a time.

`Mux`

    The pescador<2.0 `Mux` is still available and works the same,
    but is deprecated.
'''
import six
import numpy as np

from futurepast import remove, rename_parameter

from . import core
from .exceptions import PescadorError


@remove(past='1.1', future='2.0')
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

    @rename_parameter(old="seed_pool", new="streamers",
                      past='1.0', future='2.0')
    @rename_parameter(old="lam", new="rate", past='1.0', future='2.0')
    @rename_parameter(old="pool_weights", new="weights",
                      past='1.0', future='2.0')
    @rename_parameter(old='prune_empty_seeds', new='prune_empty_streams',
                      past='1.0', future='2.0')
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

        seed_pool : iterable of streamers
            .. warning:: This parameter name was deprecated in pescador 1.1
                Use the `streamers` parameter instead.
                The `seed_pool` parameter will be removed in pescador 2.0.

        lam : float > 0.0
            .. warning:: This parameter name was deprecated in pescador 1.1
                Use the `rate` parameter instead.
                The `lam` parameter will be removed in pescador 2.0.

        pool_weights : np.ndarray or None
            .. warning:: This parameter name was deprecated in pescador 1.1
                Use the `weights` parameter instead.
                The `pool_weights` parameter will be removed in pescador 2.0.

        prune_empty_seeds : bool
            .. warning:: This parameter name was deprecated in pescador 1.1
                Use the `prune_empty_streams` parameter instead.
                The `prune_empty_seeds` parameter will be removed in
                pescador 2.0.
        """
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
        with core.StreamActivator(self):

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
     * When a sample is drawn from the mux (via generate),
       chooses which active substream to stream from.
     * Handles exhaustion of streams (restarting, replacing, ...)

    """
    def __init__(self, streamers, prune_empty_streams=True,
                 random_state=None):
        """
        Parameters
        ----------
        streamers : iterable of streamers
            The collection of streamer-type objects

        prune_empty_streams : bool
            Disable streamers that produce no data. If ``True``,
            streamers that previously produced no data are never
            revisited.
            Note:
            1. This may be undesireable for streams where past emptiness
            may not imply future emptiness.
            2. [TODO: UPDATE] Failure to prune truly empty streams with
            `revive=True` can result in infinite looping behavior. Disable
            with caution.

        random_state : None, int, or np.random.RandomState
            If int, random_state is the seed used by the random number
            generator;

            If RandomState instance, random_state is the random number
            generator;

            If None, the random number generator is the RandomState instance
            used by np.random.
        """
        self.streamers = streamers
        self.n_streams = len(streamers)
        self.prune_empty_streams = prune_empty_streams

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

        # Clear state and reset actiave/deactivate params.
        self.deactivate()

    def activate(self):
        """Activates the mux as a streamer, choosing which substreams to
        select as active."""
        # These do not depend on the number of streams, k
        # This function must be filled out in more detail in a child Mux.
        self.distribution_ = 1. / self.n_streams * np.ones(self.n_streams)
        self.valid_streams_ = np.ones(self.n_streams, dtype=bool)

    def deactivate(self):
        """Reset the Mux state."""
        self.distribution_ = np.zeros(self.n_streams)
        self.valid_streams_ = np.zeros(self.n_streams)

    def iterate(self, max_iter=None):
        """Yields items from the mux."""
        if max_iter is None:
            max_iter = np.inf

        with core.StreamActivator(self):
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

                    # If we're disabling empty seeds, see if this stream
                    # produced data at any point; if it didn't, turn it off.
                    #  (Note) prune_empty_streams applies to all Muxes?
                    if (self.prune_empty_streams and
                            self.stream_counts_[idx] == 0):
                        self.distribution_[self.stream_idxs_[idx]] = 0.0
                        self.valid_streams_[self.stream_idxs_[idx]] = False

                    # Call child-class exhausted-stream behavior
                    self._on_stream_exhausted(idx)

                    # Setup a new stream for this index
                    self._replace_stream(idx)

                # If everything has been pruned, kill the while loop
                if not self.valid_streams_.any():
                    break

    def _streamers_available(self):
        "Override this to modify the behavior of the main iter loop condition."
        return True

    def _on_stream_exhausted(self, idx):
        """Override this to provide a Mux with additional behavior
        when a stream is exhausted. This gets called *after* streams
        are pruned.

        Parameters
        ----------
        idx : int, [0:k - 1]
            Index of the exhausted stream (in `self.stream_idxs_`).
        """
        pass

    def _replace_stream(self, idx):
        """Called after a stream has been exhausted, replace the stream
        with another from the pool.

        For custom behavior (weights, etc.), override in a child class.
        """
        if (self.distribution_ > 0).any():
            # Replace it and move on if there are still seeds
            # in the pool.
            self.distribution_[:] /= np.sum(self.distribution_)

            # Setup the new stream.
            self._new_stream(idx)

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
        # Choose the stream index from the candidate pool
        self.stream_idxs_[idx] = self._new_stream_index(idx)

        # Activate the Streamer, and get the weights
        self.streams_[idx] = self._activate_stream(self.stream_idxs_[idx])

        # Reset the sample count to zero
        self.stream_counts_[idx] = 0

    def _n_samples_to_stream(self):
        """Return how many samples to stream for a new streamer. None
        makes an infinite streamer. If the `BaseMux` subclass has a
        `rate` field, it would be returned here. The default - None -
        makes the resulting streamers infinite. (`max_iter`=None)
        """
        return None

    def _activate_stream(self, idx):
        '''Randomly select and create a stream.

        Parameters
        ----------
        idx : int, [0:n_streams - 1]
            The stream index to replace
        '''
        if len(self.streamers) != len(self.distribution_):
            raise PescadorError('`streamers` must have the same '
                                'length as `distribution`')

        # Get the number of samples for this streamer.
        n_stream = self._n_samples_to_stream()

        # instantiate a new streamer
        return self.streamers[idx].iterate(max_iter=n_stream)

    def _new_stream_index(self, idx=None):
        """Returns an index of a streamer from `self.streamers` which
        will get added to the active set.

        Implementation Required in any child class.

        Parameters
        ----------
        idx : int or None
            The index is passed along so a child class can use it.
            (The index is not required for a random stream as in PoissonMux,
             but would be required for RoundRobin mux).
        """
        raise NotImplementedError("_new_stream_index() must be implemented in"
                                  " a child class.")

    def _next_sample_index(self):
        """Returns the index in self.streams_ for the streamer from which
        to draw the next sample.

        Implementation required in any child class.
        """
        raise NotImplementedError("_next_sample_index() must be implemented in"
                                  " a child class.")


class WeightedStochasticMux(BaseMux):
    """A Mux which chooses streams randomly (possibly weighted).

    Expands BaseMux with the following features:
     * Adds `weights` parameter for optionally setting the weights
       of the `streams`, for modifying how often they are sampled from.

    WeightedStochasticMux is *not* a complete implementation (as it does
    not create the streams_ upon activation); you must instead use a child
    class which does this (i.e. PoissonMux, StochasticMux).
    """
    def __init__(self, streamers, weights=None,
                 prune_empty_streams=True, random_state=None):
        """
        """
        super(WeightedStochasticMux, self).__init__(
            streamers, prune_empty_streams=prune_empty_streams,
            random_state=random_state)

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

    def _new_stream_index(self, idx=None):
        """Returns a random streamer index from `self.streamers`,
        given the current distribution.
        """
        return self.rng.choice(
            self.n_streams, p=self.distribution_)

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

    def _activate_stream(self, idx):
        '''Randomly select and create a stream.

        WeightedStochasticMux adds weights to _activate_stream in addition
        to just returning the streamer.

        Parameters
        ----------
        idx : int, [0:n_streams - 1]
            The stream index to replace

        weight : float
            Present weight of the chosen streamer.
        '''
        if len(self.streamers) != len(self.weights):
            raise PescadorError('`streamers` must have the same '
                                'length as `weights`')

        streamer = super(WeightedStochasticMux, self)._activate_stream(idx)
        weight = self.weights[idx]

        return streamer, weight

    def _new_stream(self, idx):
        '''Randomly select and create a new stream.

        Parameters
        ----------
        idx : int, [0:n_streams - 1]
            The stream index to replace
        '''
        # Choose the stream index from the candidate pool
        self.stream_idxs_[idx] = self._new_stream_index(idx)

        # Activate the Streamer, and get the weights
        self.streams_[idx], self.stream_weights_[idx] = self._activate_stream(
            self.stream_idxs_[idx])

        # Reset the sample count to zero
        self.stream_counts_[idx] = 0


class PoissonMux(WeightedStochasticMux):
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
    def __init__(self, streamers, k_active,
                 rate=256.0, weights=None,
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

        if self.mode not in [
                "with_replacement", "single_active", "exhaustive"]:
            raise PescadorError("{} is not a valid mode for PoissonMux".format(
                self.mode))

        super(PoissonMux, self).__init__(
            streamers, weights=weights,
            prune_empty_streams=prune_empty_streams,
            random_state=random_state)

    def activate(self):
        # Call the parent's activate.
        super(PoissonMux, self).activate()

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
        super(PoissonMux, self).deactivate()

        self.streams_ = None
        self.stream_idxs_ = None
        self.stream_counts_ = None
        self.stream_weights_ = None
        self.weight_norm_ = None

    def _streamers_available(self):
        return self.weight_norm_ > 0.0

    def _n_samples_to_stream(self):
        "Returns rate or none."
        if self.rate is not None:
            return 1 + self.rng.poisson(lam=self.rate)
        else:
            return None

    def _next_sample_index(self):
        """PoissonMux chooses its next sample stream randomly"""
        return self.rng.choice(self.k_active,
                               p=(self.stream_weights_ /
                                  self.weight_norm_))

    def _on_stream_exhausted(self, idx):
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
        streamer, weight = super(PoissonMux, self)._activate_stream(idx)

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


class ShuffledMux(WeightedStochasticMux):
    """A variation on a mux, which takes N streamers, and samples
    from them equally, guaranteeing all N streamers to be "active",
    unlike the base Mux, which randomly chooses streams when activating.

    TODO Does this need to implement things directly, or is subclassing
    PoissonMux okay?
    """
    def __init__(self, streamers, weights=None,
                 random_state=None, prune_empty_streams=True):
        super(ShuffledMux, self).__init__(
            streamers, weights=weights,
            prune_empty_streams=prune_empty_streams,
            random_state=random_state)

    def activate(self):
        """ShuffledMux's activate is similar to PoissonMux,
        but there is no 'k_active', since all the streams are always available.
        """
        # Call the parent's activate.
        super(ShuffledMux, self).activate()

        # ShuffledMux has
        self.streams_ = [None] * self.n_streams

        # Weights of the active streams.
        # Once a stream is exhausted, it is set to 0
        self.stream_weights_ = np.zeros(self.n_streams)
        # How many samples have been drawn from each (active) stream.
        self.stream_counts_ = np.zeros(self.n_streams, dtype=int)
        # Array of pointers into `self.streamers`
        self.stream_idxs_ = np.zeros(self.n_streams, dtype=int)

        # Initialize each active stream.
        for idx in range(self.n_streams):

            if not (self.distribution_ > 0).any():
                break

            # Setup a new streamer at this index.
            self._new_stream(idx)

        self.weight_norm_ = np.sum(self.stream_weights_)

    def deactivate(self):
        super(ShuffledMux, self).deactivate()

        self.streams_ = None
        self.stream_idxs_ = None
        self.stream_counts_ = None
        self.stream_weights_ = None
        self.weight_norm_ = None

    def _streamers_available(self):
        return self.weight_norm_ > 0.0

    def _next_sample_index(self):
        """PoissonMux chooses its next sample stream randomly"""
        return self.rng.choice(self.n_streams,
                               p=(self.stream_weights_ /
                                  self.weight_norm_))

    def _on_stream_exhausted(self, idx):
        # Identical to "single_active" mode in PoissonMux
        # If we need to revive a seed, give it the max
        # current probability
        if self.distribution_.any():
            self.distribution_[self.stream_idxs_[idx]] = (
                np.max(self.distribution_))
        else:
            self.distribution_[self.stream_idxs_[idx]] = 1.0

        super(ShuffledMux, self)._on_stream_exhausted(idx)

    def _activate_stream(self, idx):
        '''Randomly select and create a stream.

        ShuffledMux always samples in "single_active" mode, so
        when a stream is activated, it's 'distribution_' is set to 0,
        so that it can't be drawn from again.

        Parameters
        ----------
        idx : int, [0:n_streams - 1]
            The stream index to replace
        '''
        streamer, weight = super(ShuffledMux, self)._activate_stream(idx)

        # Zero this stream distribution out;
        # This effectively disables this stream as soon as it is chosen,
        # preventing it from being chosen again.
        self.distribution_[idx] = 0.0

        # Correct the distribution
        if (self.distribution_ > 0).any():
            self.distribution_[:] /= np.sum(self.distribution_)

        return streamer, weight


class RoundRobinMux(BaseMux):
    """A Mux which iterates over all streamers in strict order.

    TODO: (maybe) handle stream exhaustion?

    Examples
    --------
    >>> a = pescador.Streamer("a")
    >>> b = pescador.Streamer("b")
    >>> c = pescador.Streamer("c")
    >>> mux = pescador.RoundRobinMux([a, b, c])
    >>> print("".join(mux.iterate(9)))
    "abcabcabc"
    """
    def __init__(self, streamers, random_state=None,
                 prune_empty_streams=True):
        super(RoundRobinMux, self).__init__(
            streamers,
            random_state=random_state,
            prune_empty_streams=prune_empty_streams)

    def activate(self):
        super(RoundRobinMux, self).activate()

        self.active_index_ = 0

        # The active streamers
        self.streams_ = [None] * self.n_streams

        # Stream pointers; this is used to optionally shuffle the order
        # for each complete iteration.
        self.stream_idxs_ = np.arange(self.n_streams, dtype=int)

        # How many samples have been drawn from each?
        self.stream_counts_ = np.zeros(self.n_streams, dtype=int)

        # Initialize each active stream.
        for idx in range(self.n_streams):

            if not (self.distribution_ > 0).any():
                break

            # Setup a new streamer at this index.
            self._new_stream(idx)

    def deactivate(self):
        super(RoundRobinMux, self).deactivate()
        self.active_index_ = None
        self.streams_ = None
        self.stream_idxs_ = None
        self.stream_counts_ = None

    def _new_stream_index(self, idx=None):
        return self.stream_idxs_[idx]

    def _next_sample_index(self):
        """Rotates through each active sampler by incrementing the index"""
        idx = self.active_index_
        self.active_index_ += 1
        if self.active_index_ >= len(self.streamers):
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
        # Choose the stream index from the candidate pool
        self.stream_idxs_[idx] = self._new_stream_index(idx)

        # Activate the Streamer, and get the weights
        self.streams_[idx] = self._activate_stream(self.stream_idxs_[idx])

        # Reset the sample count to zero
        self.stream_counts_[idx] = 0


class ChainMux(BaseMux):
    """As in itertools.chain(). Runs the first streamer to exhaustion,
    then the second, then the third, etc. k=1.

    Examples
    --------
    >>> a = pescador.Streamer("abc")
    >>> b = pescador.Streamer("def")
    >>> mux = pescador.mux.ChainMux([a, b], mode="exhaustive")
    >>> "".join(list(mux.iterate()))
    "abcdef"

    >>> a = pescador.Streamer("abc")
    >>> b = pescador.Streamer("def")
    >>> mux = pescador.mux.ChainMux([a, b], mode="with_replacement")
    >>> "".join(list(mux.iterate(max_iter=12)))
    "abcdefabcdef"
    """
    def __init__(self, streamers, mode="exhaustive",
                 random_state=None,
                 prune_empty_streams=True):
        """
        Parameters
        ----------
        streamers :

        mode : ["exhaustive", "with_replacement"]
            `exhaustive`
                `ChainMux will exit after each stream has been exhausted.

            `with_replacement`
                `ChainMux will restart from the beginning after each
                streamer has been run to exhaustion.
        """
        super(ChainMux, self).__init__(
            streamers, random_state=random_state,
            prune_empty_streams=prune_empty_streams)

        self.mode = mode

    def activate(self):
        # This active_index to None so the first streamer knows
        #  it hasn't been used yet.
        self.active_index_ = None
        self.completed_ = False

        super(ChainMux, self).activate()

        # Chainmux only ever has one active streamer.
        self.streams_ = [None]
        self.stream_counts_ = np.zeros(1, dtype=int)

        # Initialize the active stream.
        if (self.distribution_ > 0).any():
            # Setup a new streamer at this index.
            self._new_stream(0)

    def deactivate(self):
        super(ChainMux, self).deactivate()
        self.streams_ = None
        self.active_index_ = None
        self.stream_counts_ = None
        self.completed_ = None

    def _streamers_available(self):
        return self.completed_ is not True

    def _new_stream_index(self, idx=None):
        """Just increment the active stream every time one is requested."""
        # Streamer is starting
        if self.active_index_ is None:
            self.active_index_ = 0

        else:
            self.active_index_ += 1

        # Move to the next streamer
        if self.active_index_ >= len(self.streamers):
            self.active_index_ = 0

        return self.active_index_

    def _next_sample_index(self):
        """k==1, this is always 0."""
        return 0

    def _on_stream_exhausted(self, idx):
        # Identical to "single_active" mode in PoissonMux
        #  - ChainMux only ever operates in "single_active" mode.
        # If we need to revive a seed, give it the max
        # current probability
        if self.mode == "with_replacement":
            if self.distribution_.any():
                self.distribution_[self.active_index_] = (
                    np.max(self.distribution_))
            else:
                self.distribution_[self.active_index_] = 1.0

    def _replace_stream(self, idx):
        # If there are active streams reamining,
        # choose a new one to make active.
        if (self.distribution_ > 0).any():
            # Replace it and move on if there are still seeds
            # in the pool.
            self.distribution_[:] /= np.sum(self.distribution_)

            # Setup a new streamer at this index.
            self._new_stream(idx)
        elif self.mode == "exhaustive":
            # Otherwise, the Chain is complete.
            self.completed_ = True

    def _activate_stream(self, idx):
        '''Activate the next stream.

        Parameters
        ----------
        idx : int, [0:n_streams - 1]
            The stream index to replace
        '''
        streamer = super(ChainMux, self)._activate_stream(idx)

        # If we're sampling without replacement, zero this one out
        # This effectively disables this stream as soon as it is chosen,
        # preventing it from being chosen again (unless it is revived)
        # if not self.with_replacement:
        if self.mode != "with_replacement":
            self.distribution_[idx] = 0.0

            # Correct the distribution
            if (self.distribution_ > 0).any():
                self.distribution_[:] /= np.sum(self.distribution_)

        return streamer

    def _new_stream(self, idx):
        '''Randomly select and create a new stream.

        Parameters
        ----------
        idx : int, [0:n_streams - 1]
            The stream index to replace
        '''
        # Activate the Streamer
        self.streams_[0] = self._activate_stream(self._new_stream_index())

        # Reset the sample count to zero
        self.stream_counts_[0] = 0


"""
        '''
        Randomly select and create a new stream.

        Parameters
        ----------
        idx : int, [0:n_streams - 1]
            The stream index to replace
        '''
        # Choose the stream index from the candidate pool
        self.stream_idxs_[idx] = self._new_stream_index(idx)

        # Activate the Streamer
        self.streams_[idx] = self._activate_stream(self.stream_idxs_[idx])
        # and get the weights
        # , self.stream_weights_[idx]

        # Reset the sample count to zero
        self.stream_counts_[idx] = 0
"""
