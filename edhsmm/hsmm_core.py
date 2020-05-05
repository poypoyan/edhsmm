# HSMM (Explicit Duration HMM) Core Algorithms
# This is inspired from "_hmmc.pyx" from hmmlearn package.
# Hence, this is easily convertible to cython (with nogil).
# For cython, auxiliary functions (e.g. argmax, logsumexp,
# etc.) should be implemented like in "_hmmc.pyx".

import numpy as np
from scipy.special import logsumexp

# ctypedef double dtype_t

# forward algorithm
def _forward(n_samples, n_states, n_durations,
             log_startprob,
             log_transmat,
             log_durprob,
             log_obsprob,
             right_censor,
             eta, u, xi):
    # set number of iterations for t
    if right_censor == 0:
        t_iter = n_samples   # cdef int
    else:
        t_iter = n_samples + n_durations - 1   # cdef int
    # cdef int t, j, d, i
    # cdef dtype_t curr_u
    alpha_addends = np.empty(n_durations)
    astar_addends = np.empty(n_states)
    alpha = np.empty(n_states)
    alphastar = np.empty((t_iter, n_states))

    for j in range(n_states):
        alphastar[0, j] = log_startprob[j]
    for t in range(t_iter):
        for j in range(n_states):
            for d in range(n_durations):
                # evaluate u_t(j, d) and curr_u
                if t < n_samples:
                    if t < 1 or d < 1:
                        u[t, j, d] = log_obsprob[t, j]
                    else:
                        u[t, j, d] = u[t - 1, j, d - 1] + log_obsprob[t, j]
                    curr_u = u[t, j, d]
                elif d < t - (n_samples - 1):
                    curr_u = 0.0
                else:
                    curr_u = u[n_samples - 1, j, (n_samples - 1) + d - t]
                # alpha summation
                if t - d >= 0:
                    alpha_addends[d] = alphastar[t - d, j] + log_durprob[j, d] + curr_u
                else:
                    alpha_addends[d] = float("-inf")
                eta[t, j, d] = alpha_addends[d]   # eta initial
            alpha[j] = logsumexp(alpha_addends)
        # alphastar summation
        for j in range(n_states):
            for i in range(n_states):
                astar_addends[i] = alpha[i] + log_transmat[i, j]
                if t < n_samples:
                    xi[t, i, j] = astar_addends[i]   # xi initial
            if t < t_iter - 1:
                alphastar[t + 1, j] = logsumexp(astar_addends)

# compute for u only: this will be used by score and predict function in HSMM class
def _u_only(n_samples, n_states, n_durations,
            log_obsprob, u):
    # cdef int t, j, d
    for t in range(n_samples):
        for j in range(n_states):
            for d in range(n_durations):
                if t < 1 or d < 1:
                    u[t, j, d] = log_obsprob[t, j]
                else:
                    u[t, j, d] = u[t - 1, j, d - 1] + log_obsprob[t, j]

# backward algorithm
def _backward(n_samples, n_states, n_durations,
             log_startprob,
             log_transmat,
             log_durprob,
             log_observprob,
             right_censor,
             beta, u, betastar):
    # cdef int t, j, d, i
    bstar_addends = np.empty(n_durations)
    beta_addends = np.empty(n_states)

    for j in range(n_states):
        beta[n_samples - 1, j] = 0.0
    for t in range(n_samples - 2, -2, -1):
        for j in range(n_states):
            for d in range(n_durations):
                # betastar summation
                if t + d + 1 <= n_samples - 1:
                    bstar_addends[d] = log_durprob[j, d] + u[t + d + 1, j, d] + beta[t + d + 1, j]
                elif right_censor == 0:   # without right censor
                    bstar_addends[d] = float("-inf")
                else:   # with right censor
                    bstar_addends[d] = log_durprob[j, d] + u[n_samples - 1, j, n_samples - t - 2]
            betastar[t + 1, j] = logsumexp(bstar_addends)
        if t > -1:
            # beta summation
            for j in range(n_states):
                for i in range(n_states):
                    beta_addends[i] = log_transmat[j, i] + betastar[t + 1, i]
                beta[t, j] = logsumexp(beta_addends)

# smoothed probabilities
def _smoothed(n_samples, n_states, n_durations,
              beta, betastar,
              right_censor,
              eta, xi, gamma):
    # cdef int t, j, d, i, h
    for t in range(n_samples - 1, -1, -1):
        for i in range(n_states):
            # eta computation
            # note: if with right censoring, then eta[t, :, :] for t >= n_samples will be
            # used for gamma computation. since beta[t, :] = 0 for t >= n_samples, hence
            # no modifications to eta at t >= n_samples.
            for d in range(n_durations):
                eta[t, i, d] = eta[t, i, d] + beta[t, i]
            # xi computation
            # note: at t == n_samples - 1, if with right censoring, then xi[t, i, j] will
            # still be added with betastar[n_samples, j], but betastar[n_samples, j] = 0.
            # hence no modifications to xi[n_samples - 1, :, :].
            for j in range(n_states):
                if t == n_samples - 1 and right_censor == 0:
                    xi[t, i, j] = float("-inf")
                elif t < n_samples - 1:
                    xi[t, i, j] = xi[t, i, j] + betastar[t + 1, j]
            # gamma computation
            # note: this is the slow "original" method. the paper provides a faster
            # recursive method (using xi), but it requires subtraction and produced
            # numerical inaccuracies from our initial tests. 
            gamma[t, i] = float("-inf")
            for d in range(n_durations):
                for h in range(n_durations):
                    if h >= d and (t + d < n_samples or right_censor != 0):
                        gamma[t, i] = logsumexp([gamma[t, i], eta[t + d, i, h]])   # logaddexp

# evaluate curr_u: this will be used by viterbi algorithm below
def _curr_u(n_samples, u, t, j, d):
    if t < n_samples:
        return u[t, j, d]
    elif d < t - (n_samples - 1):
        return 0.0
    else:
        return u[n_samples - 1, j, (n_samples - 1) + d - t]

# viterbi algorithm
def _viterbi(n_samples, n_states, n_durations,
             log_startprob,
             log_transmat,
             log_duration,
             right_censor, u):
    # set number of iterations for t
    if right_censor == 0:
        t_iter = n_samples   # cdef int
    else:
        t_iter = n_samples + n_durations - 1   # cdef int
    # cdef int t, j, d, i, h
    # forward pass
    # backward pass
    pass
