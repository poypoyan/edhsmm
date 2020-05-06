import numpy as np

# masks error when applying log(0)
# copied from hmmlearn
def log_mask_zero(a):
    with np.errstate(divide="ignore"):
        return np.log(a)
