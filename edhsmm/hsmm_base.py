import numpy as np
import scipy.stats
from scipy.special import logsumexp
from sklearn import cluster
from sklearn.utils import check_array

# import hsmm_core as core
import hsmm_core_x as core # use if hsmm_core_x.pyx is compiled
from hsmm_utils import log_mask_zero

# Base Class for Explicit Duration HSMM
class HSMM:
    def __init__(self, n_states=2, n_durations=5, n_iter=20, tol=1e-2):
        if not n_states >= 2:
            raise ValueError("number of states (n_states) must be at least 2")
        if not n_durations >= 1:
            raise ValueError("number of durations (n_durations) must be at least 1")
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
                self.tmat[i, i] = 0.0   # no self-transitions in HSMM

    # check: check if properties of model parameters are satisfied
    def check(self):
        # starting probabilities
        self.pi = np.asarray(self.pi)
        if self.pi.shape != (self.n_states, ):
            raise ValueError("start probabilities (self.pi) must have shape ({},)".format(self.n_states))
        if not np.allclose(self.pi.sum(), 1.0):
            raise ValueError("start probabilities (self.pi) must add up to 1.0")
        # transition probabilities
        self.tmat = np.asarray(self.tmat)
        if self.tmat.shape != (self.n_states, self.n_states):
            raise ValueError("transition matrix (self.tmat) must have shape ({0}, {0})".format(self.n_states))
        if not np.allclose(self.tmat.sum(axis=1), 1.0):
            raise ValueError("transition matrix (self.tmat) must add up to 1.0")
        for i in range(self.n_states):
            if self.tmat[i, i] != 0.0:   # check for diagonals
                raise ValueError("transition matrix (self.tmat) must have all diagonals equal to 0.0")

    # emission_logprob: compute the log-likelihood per state of each observation
    def emission_logprob(self):
        """
        arguments: (self, X, logframe)
        return: status_code, error_info
        > build the logframe
        """
        pass   # implemented in subclass

    # emission_mstep: perform m-step for emission parameters
    def emission_mstep(self):
        """
        arguments: (self, X, gamma, lengths=None)
        return: None
        > compute the emission parameters
        """
        pass   # implemented in subclass

    # dur_logprob: compute the log-probability per state of each duration
    def dur_logprob(self):
        """
        arguments: (self, logdur)
        return: status_code, error_info
        > build the logdur
        """
        pass   # implemented in subclass

    # dur_mstep: perform m-step for duration parameters
    def dur_mstep(self):
        """
        arguments: (self, new_dur)
        return: None
        > compute the duration parameters
        """
        pass   # implemented in subclass

    # score: log-likelihood computation from observation series
    def score(self, X, lengths=None, censoring=1):
        X = check_array(X)
        self.init(X, lengths=lengths)
        self.check()
        n_samples = X.shape[0]
        # setup required probability tables
        logframe = np.empty((n_samples, self.n_states))
        logdur = np.empty((self.n_states, self.n_durations))
        beta = np.empty((n_samples, self.n_states))
        betastar = np.empty((n_samples, self.n_states))
        u = np.empty((n_samples, self.n_states, self.n_durations))
        # main computations
        dur_status, dur_info = self.dur_logprob(logdur)   # build logdur
        if dur_status == -1:
            print("SCORE: (ABORT) ", dur_info, sep="")
            return None
        emi_status, emi_info = self.emission_logprob(X, logframe)   # build logframe
        if emi_status == -1:
            print("SCORE: (ABORT) ", emi_info, sep="")
            return None
        core._u_only(n_samples, self.n_states, self.n_durations,
                     logframe, u)
        core._backward(n_samples, self.n_states, self.n_durations,
                       log_mask_zero(self.pi),
                       log_mask_zero(self.tmat),
                       logdur, censoring, beta, u, betastar)
        # compute for gamma for t = 0
        gammazero = log_mask_zero(self.pi) + betastar[0]
        return logsumexp(gammazero)   # the summation over states is the score

    # fit: parameter estimation from observation series
    def fit(self, X, lengths=None, censoring=1):
        X = check_array(X)
        self.init(X, lengths=lengths)
        self.check()
        n_samples = X.shape[0]
        # setup required probability tables
        logframe = np.empty((n_samples, self.n_states))
        logdur = np.empty((self.n_states, self.n_durations))
        beta = np.empty((n_samples, self.n_states))
        betastar = np.empty((n_samples, self.n_states))
        u = np.empty((n_samples, self.n_states, self.n_durations))
        xi = np.empty((n_samples, self.n_states, self.n_states))
        gamma = np.empty((n_samples, self.n_states))
        if censoring == 0:   # without right censoring
            eta = np.empty((n_samples, self.n_states, self.n_durations))
        else:   # with right censoring
            eta = np.empty((n_samples + self.n_durations - 1, self.n_states, self.n_durations))
        # main loop
        for itera in range(self.n_iter):
            # main computations
            dur_status, dur_info = self.dur_logprob(logdur)   # build logdur
            if dur_status == -1:
                print("FIT: (ABORT) ", dur_info, sep="")
                break
            emi_status, emi_info = self.emission_logprob(X, logframe)   # build logframe
            if emi_status == -1:
                print("FIT: (ABORT) ", emi_info, sep="")
                break
            core._u_only(n_samples, self.n_states, self.n_durations,
                         logframe, u)
            core._forward(n_samples, self.n_states, self.n_durations,
                          log_mask_zero(self.pi),
                          log_mask_zero(self.tmat),
                          logdur, censoring, eta, u, xi)
            core._backward(n_samples, self.n_states, self.n_durations,
                           log_mask_zero(self.pi),
                           log_mask_zero(self.tmat),
                           logdur, censoring, beta, u, betastar)
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
            self.pi = np.exp(gamma[0] - logsumexp(gamma[0]))
            tmat_num = logsumexp(xi, axis=0)
            self.tmat = np.exp(tmat_num - logsumexp(tmat_num, axis=1)[None].T)
            dur_num = logsumexp(eta[0:n_samples], axis=0)
            new_dur = np.exp(dur_num - logsumexp(dur_num, axis=1)[None].T)
            self.dur_mstep(new_dur)   # new durations
            self.emission_mstep(X, gamma)   # new emissions
            print("FIT: reestimation complete for ", (itera + 1), "th loop.", sep="")

    # predict: hidden state & duration estimation from observation series
    def predict(self, X, lengths=None, censoring=1):
        X = check_array(X)
        self.init(X, lengths=lengths)
        self.check()
        n_samples = X.shape[0]
        # setup required probability tables
        logframe = np.empty((n_samples, self.n_states))
        logdur = np.empty((self.n_states, self.n_durations))
        u = np.empty((n_samples, self.n_states, self.n_durations))
        # main computations
        dur_status, dur_info = self.dur_logprob(logdur)   # build logdur
        if dur_status == -1:
            print("PREDICT: (ABORT) ", dur_info, sep="")
            return None, None
        emi_status, emi_info = self.emission_logprob(X, logframe)   # build logframe
        if emi_status == -1:
            print("PREDICT: (ABORT) ", emi_info, sep="")
            return None, None
        core._u_only(n_samples, self.n_states, self.n_durations,
                     logframe, u)
        state_sequence, log_prob = core._viterbi(n_samples, self.n_states, self.n_durations,
                                                 log_mask_zero(self.pi),
                                                 log_mask_zero(self.tmat),
                                                 logdur, censoring, u)
        return state_sequence, log_prob

# Simple Gaussian Explicit Duration HSMM
class GaussianHSMM(HSMM):
    def __init__(self, n_states=2, n_durations=5, n_iter=20, tol=1e-2):
        super().__init__(n_states, n_durations, n_iter, tol)

    def init(self, X, lengths=None):
        super().init()
        if not hasattr(self, "n_dim"):   # number of dimensions
            self.n_dim = X.shape[1]
        if not hasattr(self, "dur"):   # non-parametric duration
            self.dur = np.full((self.n_states, self.n_durations), 1.0 / self.n_durations)
        if not hasattr(self, "mean"):
            kmeans = cluster.KMeans(n_clusters=self.n_states)
            kmeans.fit(X)
            self.mean = kmeans.cluster_centers_
        if not hasattr(self, "covmat"):
            # TODO: better initial covariance matrices
            self.covmat = np.repeat(np.identity(self.n_dim)[None], self.n_states, axis=0)

    def check(self):
        super().check()
        # duration probabilities
        self.dur = np.asarray(self.dur)
        if self.dur.shape != (self.n_states, self.n_durations):
            raise ValueError("duration probabilities (self.dur) must have shape ({}, {})"
                             .format(self.n_states, self.n_durations))
        if not np.allclose(self.dur.sum(axis=1), 1.0):
            raise ValueError("duration probabilities (self.dur) must add up to 1.0")
        # means
        self.mean = np.asarray(self.mean)
        if self.mean.shape != (self.n_states, self.n_dim):
            raise ValueError("means (self.mean) must have shape ({}, {})"
                             .format(self.n_states, self.n_dim))
        # covariance matrices
        self.covmat = np.asarray(self.covmat)
        if self.covmat.shape != (self.n_states, self.n_dim, self.n_dim):
            raise ValueError("covariance matrices (self.covmat) must have shape ({0}, {1}, {1})"
                             .format(self.n_states, self.n_dim))

    def dur_logprob(self, logdur):
        # non-parametric duration
        logdur[:] = log_mask_zero(self.dur)
        return 0, "OK"

    def dur_mstep(self, new_dur):
        # non-parametric duration
        self.dur = new_dur
        
    def emission_logprob(self, X, logframe):
        # status: abort EM loop if any covariance matrix is not symmetric, positive-definite.
        # adapted from hmmlearn 0.2.3
        for n, cv in enumerate(self.covmat):
            if (not np.allclose(cv, cv.T) or np.any(np.linalg.eigvalsh(cv) <= 0)):
                return -1, "component {} of covariance matrix is not symmetric, positive-definite.".format(n)
                # https://www.youtube.com/watch?v=tWoFaPwbzqE&t=1694s
        n_samples = X.shape[0]
        for i in range(self.n_states):
            multigauss = scipy.stats.multivariate_normal(self.mean[i], self.covmat[i])
            for j in range(n_samples):
                logframe[j, i] = log_mask_zero(multigauss.pdf(X[j]))
        return 0, "OK"

    def emission_mstep(self, X, gamma, lengths=None):
        # NOTE: gamma is still in logarithm form
        denominator = logsumexp(gamma[None].T, axis=1)
        weight_normalized = np.exp(gamma[None].T - denominator[:, None])
        # compute means (from definition; weighted)
        self.mean = (weight_normalized * X).sum(1)
        # compute covariance matrices (from definition; weighted)
        dist = X - self.mean[:, None]
        self.covmat = ((dist * weight_normalized)[:, :, :, None] * dist[:, :, None]).sum(1)