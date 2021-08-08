import math
from scipy.stats import norm
from itertools import chain, combinations
import numpy as np
import sys

import plot_results

def non_empty_powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def nth(l, n):
    """
    Returns only nth elemnt in a list.
    """
    return [a[n] for a in l]

def holm(avgranks, N, names):
    p_values = pvalues(avgranks, N)
    k = len(avgranks)
    m = k*(k-1) / 2

    apv = []

    pv_list = nth(p_values, 0)
    for i, (_, (alg1, alg2)) in enumerate(p_values):
        v = m * pv_list[0]
        for j in range(1, i+1):
            v = max(v, (m - j)*pv_list[j])

        apv.append((min(v, 1), (alg1, alg2)))

    mon_apv = correct_monotocity(apv)
    apv_names = [(p, (names[alg1], names[alg2])) for (p, (alg1, alg2)) in mon_apv]
    return apv_names

def bh_exhaustivesets(classifiers):
    e = set()
    k = len(classifiers)

    i_full = set()
    for i in range(k-1):
        for j in range(i+1, k):
            i_full.add((classifiers[i], classifiers[j]))

    if not i_full:
        return frozenset()

    e.add(frozenset(i_full))

    classifier_set = set(classifiers)
    for p in non_empty_powerset(classifiers[:-1]):
        c1 = list(p)
        c2 = sorted(list(classifier_set - set(c1)))

        e1 = bh_exhaustivesets(c1)
        e2 = bh_exhaustivesets(c2)

        e = e.union(e1)
        e = e.union(e2)

        for f1 in e1:
            for f2 in e2:
                e.add(f1.union(f2))

    if frozenset() in e:
        e.remove(frozenset())
    return frozenset(e)

def min_pvalue_set(exhaustive_set, pvalues_dict):
    m = sys.float_info.max
    for h in exhaustive_set:
        m = min(m, pvalues_dict[h])
    return m

def correct_monotocity(p_values):
    new_pvalues = p_values.copy()
    array_pvalues = np.asarray([p[0] for p in p_values])

    for i, (_, h) in enumerate(p_values[1:]):
        i = i + 1
        new_pvalues[i] = (array_pvalues[:i+1].max(), h)

    return new_pvalues

def bergmann_hommel(avgranks, N, names):
    p_values = pvalues(avgranks, N)

    k = len(avgranks)
    exhaustive_sets = bh_exhaustivesets(np.arange(k))
    pvalues_dict = {h: p for (p, h) in p_values}

    apv = []
    for i, (_, hypot) in enumerate(p_values):
        v = sys.float_info.min
        for ex_set in exhaustive_sets:
            if hypot in ex_set:
                v = max(v, len(ex_set)*
                        min_pvalue_set(ex_set, pvalues_dict))

        apv.append((min(v, 1), hypot))

    mon_apv = correct_monotocity(apv)
    apv_names = [(p, (names[alg1], names[alg2])) for (p, (alg1, alg2)) in mon_apv]
    return apv_names

def pvalues(avgranks, N):
    k = len(avgranks)

    z_values = {}
    p_values = []
    for i in range(k-1):
        for j in range(i+1, k):
            z_values[(i,j)] = abs((avgranks[i] - avgranks[j]) / (math.sqrt(k*(k+1)/(6*N))))

            p_value = (1 - norm.cdf(z_values[(i,j)]))*2
            p_values.append((p_value, (i, j)))

    sort_pvalues = sorted(p_values)

    return sort_pvalues

def print_rejected_pvalues(apv, alpha=0.05):

    for (p, (alg1, alg2)) in apv:
        if p < alpha:
            print("Rejected " + str(alg1) + " vs " + str(alg2) + ". p-value = " + str(p))

if __name__ == "__main__":
    # avgranks = [2.1, 3.25, 2.2, 4.333, 3.117]
    # names = ["C4.5", "1-NN", "NaiveBayes", "Kernel", "CN2"]

    df_algorithms = plot_results.read_results()
    rank = df_algorithms.rank(axis=1, ascending=False)

    avgranks = rank.mean().to_numpy()
    names = rank.columns.values

    print()
    print("Bergmann Hommel")
    print("--------------------")
    apv = bergmann_hommel(avgranks, rank.shape[0], names)

    print(apv)
    print()
    print("Rejected")
    print("--------------------")
    print_rejected_pvalues(apv)

    print()
    print("Holm")
    print("--------------------")
    apv = holm(avgranks, rank.shape[0], names)

    print(apv)
    print()
    print("Rejected")
    print("--------------------")
    print_rejected_pvalues(apv)


    # exhaustive_sets = bh_exhaustivesets([1, 2, 3, 4, 5])
    # print(str(len(exhaustive_sets)) + " sets")
    # for s in exhaustive_sets:
    #     print("(", end="")
    #     for it in s:
    #         print(str(it[0]) + str(it[1]) + ",", end="")
    #     print(")")

