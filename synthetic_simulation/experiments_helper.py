import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.linalg import solve_triangular
from scipy.stats import norm
from pybnesian.factors import NodeType

TESTS = ["LinearCorrelation", "RCoT"]
INSTANCES = [200, 2000, 10000]
TRAINING_FOLDS = [10]
PATIENCE = [0, 5]
SEED=0
NUM_SIMULATIONS=50

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

def shd(estimated, true):
    assert set(estimated.nodes()) == set(true.nodes())
    shd_value = 0

    estimated_arcs = set(estimated.arcs())
    true_arcs = set(true.arcs())

    for est_arc in estimated_arcs:
        if est_arc not in true.arcs():
            shd_value += 1
            s, d = est_arc
            if (d, s) in true_arcs:
                true_arcs.remove((d, s))

    for true_arc in true_arcs:
        if true_arc not in estimated_arcs:
            shd_value += 1

    return shd_value

def hamming(estimated, true):
    assert set(estimated.nodes()) == set(true.nodes())
    hamming_value = 0

    estimated_arcs = set(estimated.arcs())
    true_arcs = set(true.arcs())

    for est_arc in estimated_arcs:
        s, d = est_arc
        if (s, d) not in true_arcs and (d, s) not in true_arcs:
            hamming_value += 1

    for true_arc in true_arcs:
        s, d = true_arc
        if (s, d) not in estimated_arcs and (d, s) not in estimated_arcs:
            hamming_value += 1

    return hamming_value

def hamming_type(estimated, true):
    assert set(estimated.nodes()) == set(true.nodes())
    hamming_value = 0

    for n in true.nodes():
        if estimated.node_type(n) != true.node_type(n):
            hamming_value += 1

    return hamming_value