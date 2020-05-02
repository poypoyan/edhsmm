import numpy as np

# copied from hmmlearn
def log_mask_zero(a):
    """Computes the log of input probabilities masking divide by zero in log.

    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalized to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.

    This function masks this unharmful warning.
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        return np.log(a)