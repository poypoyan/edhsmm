import numpy as np
from scipy.special import logsumexp

from . import hsmm_base, hsmm_utils
from .hsmm_base import HSMM
from .hsmm_utils import log_mask_zero


# Explicit Duration HSMM with Multinomial (Discrete) Emissions
class MultinomialHSMM(HSMM):
    def __init__(self, n_states=2, n_durations=5, n_iter=20, tol=1e-2, random_state=None):
        super().__init__(n_states, n_durations, n_iter, tol, random_state)

    def _init(self, X):
        super()._init()
        # note for programmers: for every attribute that needs X in score()/predict()/fit(),
        # there must be a condition "if X is None" because sample() doesn't need an X, but
        # default attribute values must be initiated for sample() to proceed.
        if not hasattr(self, "emit"):   # also set self.n_symbols here
            if X is None:   # default for sample()
                self.n_symbols = 2
            else:
                self.n_symbols = np.max(X) + 1
            # like in hmmlearn, whether with X or not, default self.emit would be random
            rnd_checked = np.random.default_rng(self.random_state)
            init_emit = rnd_checked.random((self.n_states, self.n_symbols))
            # normalize probabilities, and make sure we don't divide by zero
            init_sum = init_emit.sum(1)
            zero_sums = (init_sum == 0)   # which rows are all zeros?
            init_emit[zero_sums] = 1   # set all rows with all zeros to all ones
            init_sum[zero_sums] = self.n_symbols
            self.emit = init_emit / init_sum[None].T
        else:
            self.n_symbols = self.emit.shape[1]   # also default for sample()

    def _check(self):
        super()._check()
        # emission probabilities
        self.emit = np.asarray(self.emit)
        if self.emit.shape != (self.n_states, self.n_symbols):
            raise ValueError("emission probabilities (self.emit) must have shape ({}, {})"
                             .format(self.n_states, self.n_symbols))
        if not np.allclose(self.emit.sum(axis=1), 1.0):
            raise ValueError("emission probabilities (self.emit) must add up to 1.0")

    def _dur_init(self):
        # non-parametric duration
        if not hasattr(self, "dur"):
            self.dur = np.full((self.n_states, self.n_durations), 1.0 / self.n_durations)

    def _dur_check(self):
        self.dur = np.asarray(self.dur)
        if self.dur.shape != (self.n_states, self.n_durations):
            raise ValueError("duration probabilities (self.dur) must have shape ({}, {})"
                             .format(self.n_states, self.n_durations))
        if not np.allclose(self.dur.sum(axis=1), 1.0):
            raise ValueError("duration probabilities (self.dur) must add up to 1.0")

    def _dur_probmat(self):
        # non-parametric duration
        return self.dur

    def _dur_mstep(self, new_dur):
        # non-parametric duration
        self.dur = new_dur

    def _emission_logl(self, X):
        return log_mask_zero(self.emit[:, np.concatenate(X)].T)

    def _emission_mstep(self, X, emission_var):
        denominator = logsumexp(emission_var, axis=0)
        weight_normalized = np.exp(emission_var - denominator)
        iverson = (X.T == np.arange(self.n_symbols)[:,None])   # iverson bracket
        self.emit = (weight_normalized[:,:,None] * iverson[:,None].T).sum(0)

    def _state_sample(self, state, random_state=None):
        emit_cdf = np.cumsum(self.emit[state, :])
        rnd_checked = np.random.default_rng(random_state)
        return [(emit_cdf > rnd_checked.random()).argmax()]   # shape of X must be (n_samples, 1)