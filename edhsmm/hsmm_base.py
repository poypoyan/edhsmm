import numpy as np
import scipy.stats
from scipy.special import logsumexp

import hsmm_core as core
from hsmm_utils import log_mask_zero

# Base Class for Explicit Duration HSMM
class HSMM:
    def __init__(self, n_states=2, n_durations=5, n_iter=20, tol=1e-2):
        self.n_states = n_states
        self.n_durations = n_durations
        self.n_iter = n_iter
        self.tol = tol
    # init: initializes model parameters if there are none yet.
    def init(self):
        if not hasattr(self, "pi"):
            self.pi = np.full(self.n_states, 1.0 / self.n_states)
        if not hasattr(self, "tmat"):
            self.tmat = np.full((self.n_states, self.n_states), 1.0 / (self.n_states - 1))
            for i in range(self.n_states):
                self.tmat[i, i] = 0.0
        if not hasattr(self, "dur"):
            self.dur = np.full((self.n_states, self.n_durations), 1.0 / self.n_durations)
    # check: check if properties of model parameters are satisfied
    def check(self):
        pass   # lazy for this
    # emission_logprob: compute the log-likelihood per state of each observation
    def emission_logprob(self, X):
        # note: subclass will implement this.
        # it should *return* the logframe.
        pass
    # emission_mstep: perform m-step on emission parameters
    def emission_mstep(self, X, gamma, lengths=None):
        # note: subclass will implement this.
        pass
    # score: compute the log-likelihood of the whole observation series
    def score(self, X, lengths=None, censoring=1):
        self.check()
        logframe = self.emission_logprob(X)
        n_samples = logframe.shape[0]
        # setup required probability tables
        beta = np.zeros((n_samples, self.n_states))
        betastar = np.zeros((n_samples, self.n_states))
        u = np.empty((n_samples, self.n_states, self.n_durations))
        # core stuff
        core._u_only(n_samples, self.n_states, self.n_durations,
                     logframe, u)
        core._backward(n_samples, self.n_states, self.n_durations,
                      log_mask_zero(self.pi),
                      log_mask_zero(self.tmat),
                      log_mask_zero(self.dur),
                      censoring, beta, u, betastar)
        # compute for gamma for t=0
        gammazero = log_mask_zero(self.pi) + betastar[0, :]
        return logsumexp(gammazero)   # the summation over states is the score
    # fit
    def fit(self, X, lengths=None, censoring=1):
        self.check()
        logframe = self.emission_logprob(X)
        n_samples = logframe.shape[0]
        # setup required probability tables
        beta = np.zeros((n_samples, self.n_states))
        betastar = np.zeros((n_samples, self.n_states))
        u = np.empty((n_samples, self.n_states, self.n_durations))
        xi = np.empty((n_samples, self.n_states, self.n_states))
        gamma = np.zeros((n_samples, self.n_states))
        if censoring == 0:   # without right censoring
            eta = np.empty((n_samples, self.n_states, self.n_durations))
        else:   # with right censoring
            eta = np.empty((n_samples + self.n_durations - 1, self.n_states, self.n_durations))
        # main loop
        for itera in range(self.n_iter):
            # core stuff
            core._u_only(n_samples, self.n_states, self.n_durations,
                         logframe, u)
            core._forward(n_samples, self.n_states, self.n_durations,
                      log_mask_zero(self.pi),
                      log_mask_zero(self.tmat),
                      log_mask_zero(self.dur),
                      censoring, eta, u, xi)
            core._backward(n_samples, self.n_states, self.n_durations,
                      log_mask_zero(self.pi),
                      log_mask_zero(self.tmat),
                      log_mask_zero(self.dur),
                      censoring, beta, u, betastar)
            core._smoothed(n_samples, self.n_states, self.n_durations,
                       beta, betastar, censoring, eta, xi, gamma)
            # check for loop break
            score = logsumexp(gamma[0, :])   # this is the output of 'score' function
            if itera > 0 and (score - old_score) < self.tol:
                print("FIT: converged at ", (itera + 1), "th loop.", sep="")
                break
            else:
                old_score = score
            # reestimation / M-step
            eta = np.exp(eta)
            xi = np.exp(xi)
            gamma = np.exp(gamma)
            self.pi = gamma[0] / gamma[0].sum()
            tmat_num = xi.sum(0)
            self.tmat = tmat_num / tmat_num.sum(1)[None].T
            dur_num = eta[0 : n_samples].sum(0)
            self.dur = dur_num / dur_num.sum(1)[None].T
            self.emission_mstep(X, gamma)   # new emissions
            print("FIT: reestimation complete for ", (itera + 1), "th loop.", sep="")
    # predict
    def predict(self, X, lengths=None, censoring=1):
        self.check()
        logframe = self.emission_logprob(X)
        n_samples = logframe.shape[0]
        # setup required probability tables
        u = np.empty((n_samples, self.n_states, self.n_durations))
        # core stuff
        core._u_only(n_samples, self.n_states, self.n_durations,
                     logframe, u)
        log_prob, state_sequence = core._viterbi(n_samples, self.n_states, self.n_durations,
                                                 log_mask_zero(self.pi),
                                                 log_mask_zero(self.tmat),
                                                 log_mask_zero(self.dur),
                                                 censoring, u)
        return log_prob, state_sequence

# Simple Gaussian Explicit Duration HSMM
class GaussianHSMM(HSMM):
    def __init__(self, n_states=1, n_durations=5, n_iter=20, tol=1e-2):
        super().__init__(n_states, n_durations, n_iter, tol)
    def init(self, X):
        super().init()
        if not hasattr(self, "means"):
            # rhis for now because lazy. in practice, use K-means
            self.means = np.full(self.n_states, 0.0)
        if not hasattr(self, "sdev"):
            self.sdev = np.full(self.n_states, 1.0)
    def emission_logprob(self, X):
        series = np.asarray(X)
        n_samples = series.shape[0]
        frame = np.empty((n_samples, self.n_states))
        for i in range(self.n_states):
            gauss = scipy.stats.norm(self.means[i], self.sdev[i])
            for j in range(n_samples):
                frame[j, i] = np.log(gauss.pdf(series[j]))
        return frame
    def emission_mstep(self, X, gamma, lengths=None):
        # based from hsmmlearn
        denominator = gamma.sum(0)
        self.means = (gamma * X[None].T).sum(0) / denominator
        self.sdev = np.sqrt((gamma * ((X - self.means[:, None]) ** 2).T).sum(0) / denominator)
