import numpy as np


def kl_divergence(p, log_p, log_q):
    kl_div = np.sum(p * (log_p - log_q))
    return kl_div

