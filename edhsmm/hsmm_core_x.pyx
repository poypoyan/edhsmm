# cython: language_level=3, boundscheck=False, wraparound=False

import numpy as np

cdef extern from "math.h":
    long double expl(long double) nogil
    long double logl(long double) nogil
    long double log1pl(long double) nogil
    int isinf(long double) nogil
    long double fabsl(long double) nogil
    const float INFINITY

ctypedef double dtype_t

# auxiliary math functions: copied from hmmlearn 0.2.3

cdef inline int _argmax(dtype_t[:] X) nogil:
    cdef dtype_t X_max = -INFINITY
    cdef int pos = 0
    cdef int i
    for i in range(X.shape[0]):
        if X[i] > X_max:
            X_max = X[i]
            pos = i
    return pos

cdef inline dtype_t _max(dtype_t[:] X) nogil:
    return X[_argmax(X)]

cdef inline dtype_t _logsumexp(dtype_t[:] X) nogil:
    cdef dtype_t X_max = _max(X)
    if isinf(X_max):
        return -INFINITY

    cdef dtype_t acc = 0
    for i in range(X.shape[0]):
        acc += expl(X[i] - X_max)
    return logl(acc) + X_max

cdef inline dtype_t _logaddexp(dtype_t a, dtype_t b) nogil:
    if isinf(a) and a < 0:
        return b
    elif isinf(b) and b < 0:
        return a
    else:
        return max(a, b) + log1pl(expl(-fabsl(a - b)))

# main functions

# compute for u_t(j, d)
def _u_only(int n_samples, int n_states, int n_durations,
            dtype_t[:, :] log_obsprob, dtype_t[:, :, :] u):
    cdef int t, j, d

    with nogil:
        for t in range(n_samples):
            for j in range(n_states):
                for d in range(n_durations):
                    if t < 1 or d < 1:
                        u[t, j, d] = log_obsprob[t, j]
                    else:
                        u[t, j, d] = u[t - 1, j, d - 1] + log_obsprob[t, j]

# evaluate current u_t(j, d). extends to t > n_samples - 1.
cdef inline dtype_t _curr_u(int n_samples, dtype_t[:, :, :] u, int t, int j, int d) nogil:
    if t < n_samples:
        return u[t, j, d]
    elif d < t - (n_samples - 1):
        return 0.0
    else:
        return u[n_samples - 1, j, (n_samples - 1) + d - t]

# forward algorithm
def _forward(int n_samples, int n_states, int n_durations,
             dtype_t[:] log_startprob,
             dtype_t[:, :] log_transmat,
             dtype_t[:, :] log_durprob,
             int right_censor,
             dtype_t[:, :, :] eta, dtype_t[:, :, :] u, dtype_t[:, :, :] xi):
    cdef int t_iter, t, j, d, i
    # set number of iterations for t
    if right_censor == 0:
        t_iter = n_samples
    else:
        t_iter = n_samples + n_durations - 1
    cdef dtype_t[::1] alpha_addends = np.empty(n_durations)
    cdef dtype_t[::1] astar_addends = np.empty(n_states)
    cdef dtype_t[::1] alpha = np.empty(n_states)
    cdef dtype_t[:, ::1] alphastar = np.empty((t_iter, n_states))

    with nogil:
        for j in range(n_states):
            alphastar[0, j] = log_startprob[j]
        for t in range(t_iter):
            for j in range(n_states):
                for d in range(n_durations):
                    # alpha summation
                    if t - d >= 0:
                        alpha_addends[d] = alphastar[t - d, j] + log_durprob[j, d] + \
                                           _curr_u(n_samples, u, t, j, d)
                    else:
                        alpha_addends[d] = -INFINITY
                    eta[t, j, d] = alpha_addends[d]   # eta initial
                alpha[j] = _logsumexp(alpha_addends)
            # alphastar summation
            for j in range(n_states):
                for i in range(n_states):
                    astar_addends[i] = alpha[i] + log_transmat[i, j]
                    if t < n_samples:
                        xi[t, i, j] = astar_addends[i]   # xi initial
                if t < t_iter - 1:
                    alphastar[t + 1, j] = _logsumexp(astar_addends)

# backward algorithm
def _backward(int n_samples, int n_states, int n_durations,
             dtype_t[:] log_startprob,
             dtype_t[:, :] log_transmat,
             dtype_t[:, :] log_durprob,
             int right_censor,
             dtype_t[:, :] beta, dtype_t[:, :, :] u, dtype_t[:, :] betastar):
    cdef int t, j, d, i
    cdef dtype_t[::1] bstar_addends = np.empty(n_durations)
    cdef dtype_t[::1] beta_addends = np.empty(n_states)

    with nogil:
        for j in range(n_states):
            beta[n_samples - 1, j] = 0.0
        for t in range(n_samples - 2, -2, -1):
            for j in range(n_states):
                for d in range(n_durations):
                    # betastar summation
                    if t + d + 1 <= n_samples - 1:
                        bstar_addends[d] = log_durprob[j, d] + u[t + d + 1, j, d] + beta[t + d + 1, j]
                    elif right_censor == 0:   # without right censor
                        bstar_addends[d] = -INFINITY
                    else:   # with right censor
                        bstar_addends[d] = log_durprob[j, d] + u[n_samples - 1, j, n_samples - t - 2]
                betastar[t + 1, j] = _logsumexp(bstar_addends)
            if t > -1:
                # beta summation
                for j in range(n_states):
                    for i in range(n_states):
                        beta_addends[i] = log_transmat[j, i] + betastar[t + 1, i]
                    beta[t, j] = _logsumexp(beta_addends)

# smoothed probabilities
def _smoothed(int n_samples, int n_states, int n_durations,
              dtype_t[:, :] beta, dtype_t[:, :] betastar,
              int right_censor,
              dtype_t[:, :, :] eta, dtype_t[:, :, :] xi, dtype_t[:, :] gamma):
    cdef int t, j, d, i, h

    with nogil:
        for t in range(n_samples - 1, -1, -1):
            for i in range(n_states):
                # eta computation
                # note: if with right censoring, then eta[t, :, :] for t >= n_samples will be
                # used for gamma computation. since beta[t, :] = 0 for t >= n_samples, hence
                # no modifications to eta at t >= n_samples.
                for d in range(n_durations):
                    eta[t, i, d] = eta[t, i, d] + beta[t, i]
                # xi computation
                # note: at t == n_samples - 1, it is decided that xi[t, :, :] should be log(0),
                # either with right censoring or without, because there is no more next data.
                for j in range(n_states):
                    if t == n_samples - 1:
                        xi[t, i, j] = -INFINITY
                    else:
                        xi[t, i, j] = xi[t, i, j] + betastar[t + 1, j]
                # gamma computation
                # note: this is the slow "original" method. the paper provides a faster
                # recursive method (using xi), but it requires subtraction and produced
                # numerical inaccuracies from our initial tests. 
                gamma[t, i] = -INFINITY
                for d in range(n_durations):
                    for h in range(n_durations):
                        if h >= d and (t + d < n_samples or right_censor != 0):
                            gamma[t, i] = _logaddexp(gamma[t, i], eta[t + d, i, h])   # logaddexp

# viterbi algorithm
def _viterbi(int n_samples, int n_states, int n_durations,
             dtype_t[:] log_startprob,
             dtype_t[:, :] log_transmat,
             dtype_t[:, :] log_duration,
             int right_censor, dtype_t[:, :, :] u):
    cdef int t_iter, t, j, d, i, j_dur, back_state, back_dur, back_t
    cdef dtype_t log_prob
    # set number of iterations for t
    if right_censor == 0:
        t_iter = n_samples
    else:
        t_iter = n_samples + n_durations - 1
    cdef dtype_t[:, ::1] delta = np.empty((t_iter, n_states))
    cdef int[:, :, ::1] psi = np.empty((t_iter, n_states, 2), dtype=np.int32)
    cdef dtype_t[::1] buffer0 = np.empty(n_states)
    cdef dtype_t[::1] buffer1 = np.empty(n_durations)
    cdef int[::1] buffer1_state = np.empty(n_durations, dtype=np.int32)
    cdef int[::1] state_sequence = np.empty(n_samples, dtype=np.int32)

    with nogil:
        # forward pass
        for t in range(t_iter):
            for j in range(n_states):
                for d in range(n_durations):
                    if t - d == 0:   # beginning
                        buffer1[d] = log_startprob[j] + log_duration[j, d] + \
                                      _curr_u(n_samples, u, t, j, d)
                        buffer1_state[d] = -1   # place-holder only
                    elif t - d > 0:   # ongoing
                        for i in range(n_states):
                            if i != j:
                                buffer0[i] = delta[t - d - 1, i] + log_transmat[i, j] + \
                                             _curr_u(n_samples, u, t, j, d)         
                            else:
                                buffer0[i] = -INFINITY
                        buffer1[d] = _max(buffer0) + log_duration[j, d]
                        buffer1_state[d] = _argmax(buffer0)
                    else:   # this should not be chosen
                        buffer1[d] = -INFINITY
                delta[t, j] = _max(buffer1)        
                j_dur = _argmax(buffer1)
                psi[t, j, 0] = j_dur   # psi[:, j, 0] is the duration of j
                psi[t, j, 1] = buffer1_state[j_dur]   # psi[:, j, 1] is the state leading to j
        # getting the last state and maximum log probability
        if right_censor == 0:
            log_prob = _max(delta[n_samples - 1])
            back_state = _argmax(delta[n_samples - 1])
            back_dur = psi[n_samples - 1, back_state, 0]
        else:
            for d in range(n_durations):
                buffer1[d] = _max(delta[n_samples + d - 1])
                buffer1_state[d] = _argmax(delta[n_samples + d - 1])
            log_prob = _max(buffer1)
            j_dur = _argmax(buffer1)
            back_state = buffer1_state[j_dur]
            back_dur = psi[n_samples + j_dur - 1, back_state, 0] - j_dur
        # backward pass
        back_t = n_samples - 1
        for t in range(n_samples - 1, -1, -1):
            if back_dur < 0:
                back_state = psi[back_t, back_state, 1]
                back_dur = psi[t, back_state, 0]
                back_t = t
            state_sequence[t] = back_state
            back_dur -= 1

    return state_sequence, log_prob