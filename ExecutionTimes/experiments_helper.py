import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.linalg import solve_triangular
from scipy.stats import norm

TRAINING_FOLDS = [10]
PATIENCE = [0, 5]
SEED=0
INSTANCES = [200, 500, 1000, 2000, 4000, 10000]

def sample_mixture(prior_prob, means, variances, n_instances, seed=0):
    np.random.seed(seed)
    p = np.asarray(prior_prob)
    c = np.cumsum(p)
    m = np.asarray(means)
    v = np.asarray(variances)

    s = np.random.uniform(size=n_instances)

    digitize = np.digitize(s, c)

    res = np.random.normal(m[digitize], np.sqrt(v[digitize]))

    return res
