import numpy as np

# iterator for X with multiple observation sequences
# copied from hmmlearn
def iter_from_X_lengths(X, lengths):
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] != n_samples:
            raise ValueError("{:d} samples do not match lengths array {!s}"
                             .format(n_samples, lengths))

        for i in range(len(lengths)):
            yield start[i], end[i]

# masks error when applying log(0)
# copied from hmmlearn
def log_mask_zero(a):
    with np.errstate(divide="ignore"):
        return np.log(a)