import os
import glob
import numpy as np
import pandas as pd
import experiments_helper
import pathlib
from pybnesian.learning.algorithms import GreedyHillClimbing
from pybnesian.learning.algorithms.callbacks import SaveModel
from pybnesian.learning.operators import ArcOperatorSet
from pybnesian.learning.scores import BIC
from pybnesian.models import GaussianNetwork
import time

patience = experiments_helper.PATIENCE

small_results = pd.DataFrame()
medium_results = pd.DataFrame()
large_results = pd.DataFrame()

for n in experiments_helper.INSTANCES:
    df = pd.read_csv('data/small_' + str(n) + ".csv")

    executions = np.empty((20000,))
    for i in range(20000):
        if i % 10 == 0:
            print(str(i) + " executions")
        bic = BIC(df)
        start_model = GaussianNetwork(list(df.columns.values))
        hc = GreedyHillClimbing()
        arcs = ArcOperatorSet()

        start = time.time()
        bn = hc.estimate(arcs, bic, start_model)
        end = time.time()

        executions[i] = end - start

    small_results['GBN_BIC_' + str(n)] = pd.Series(executions, name="GBN_BIC_" + str(n))
    print("Small " + str(n) + " -- Time: " + str(executions.mean()) + ", std: " + str(np.std(executions, ddof=1)))

    df = pd.read_csv('data/medium_' + str(n) + ".csv")

    executions = np.empty((20000,))
    for i in range(20000):
        print(str(i) + " executions")
        bic = BIC(df)
        start_model = GaussianNetwork(list(df.columns.values))
        hc = GreedyHillClimbing()
        arcs = ArcOperatorSet()

        start = time.time()
        bn = hc.estimate(arcs, bic, start_model)
        end = time.time()

        executions[i] = end - start

    medium_results['GBN_BIC_' + str(n)] = pd.Series(executions, name="GBN_BIC_" + str(n))
    print("Medium " + str(n) + " -- Time: " + str(executions.mean()) + ", std: " + str(np.std(executions, ddof=1)))

    df = pd.read_csv('data/large_' + str(n) + ".csv")

    executions = np.empty((20000,))
    for i in range(20000):
        print(str(i) + " executions")
        bic = BIC(df)
        start_model = GaussianNetwork(list(df.columns.values))
        hc = GreedyHillClimbing()
        arcs = ArcOperatorSet()

        start = time.time()
        bn = hc.estimate(arcs, bic, start_model)
        end = time.time()

        executions[i] = end - start

    large_results['GBN_BIC_' + str(n)] = pd.Series(executions, name="GBN_BIC_" + str(n))
    print("Large " + str(n) + " -- Time: " + str(executions.mean()) + ", std: " + str(np.std(executions, ddof=1)))

small_results.to_csv("HC_GBN_BIC_small.csv", index=False)
medium_results.to_csv("HC_GBN_BIC_medium.csv", index=False)
large_results.to_csv("HC_GBN_BIC_large.csv", index=False)