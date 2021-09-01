import os
import glob
import numpy as np
import pandas as pd
import experiments_helper
import pathlib
from pybnesian.learning.algorithms import GreedyHillClimbing
from pybnesian.learning.algorithms.callbacks import SaveModel
from pybnesian.learning.operators import ArcOperatorSet
from pybnesian.learning.scores import ValidatedLikelihood
from pybnesian.models import KDENetwork
import time

patience = experiments_helper.PATIENCE

small_results = pd.DataFrame()
medium_results = pd.DataFrame()
large_results = pd.DataFrame()

for n in experiments_helper.INSTANCES:
    df = pd.read_csv('data/small_' + str(n) + ".csv")

    executions = np.empty((200,))
    for i in range(200):
        if i % 10 == 0:
            print(str(i) + " executions")
        vl = ValidatedLikelihood(df, k=10, seed=i)
        start_model = KDENetwork(list(df.columns.values))
        hc = GreedyHillClimbing()
        arcs = ArcOperatorSet()

        start = time.time()
        bn = hc.estimate(arcs, vl, start_model, patience=0)
        end = time.time()

        executions[i] = end - start

    small_results['KDEBN_' + str(n)] = pd.Series(executions, name="KDEBN_" + str(n))
    print("Small " + str(n) + " -- Time: " + str(executions.mean()) + ", std: " + str(np.std(executions, ddof=1)))

    df = pd.read_csv('data/medium_' + str(n) + ".csv")

    executions = np.empty((50,))
    for i in range(50):
        print(str(i) + " executions")
        vl = ValidatedLikelihood(df, k=10, seed=i)
        start_model = KDENetwork(list(df.columns.values))
        hc = GreedyHillClimbing()
        arcs = ArcOperatorSet()

        start = time.time()
        bn = hc.estimate(arcs, vl, start_model, patience=0)
        end = time.time()

        executions[i] = end - start

    medium_results['KDEBN_' + str(n)] = pd.Series(executions, name="KDEBN_" + str(n))
    print("Medium " + str(n) + " -- Time: " + str(executions.mean()) + ", std: " + str(np.std(executions, ddof=1)))

    df = pd.read_csv('data/large_' + str(n) + ".csv")

    executions = np.empty((20,))
    for i in range(20):
        print(str(i) + " executions")
        vl = ValidatedLikelihood(df, k=10, seed=i)
        start_model = KDENetwork(list(df.columns.values))
        hc = GreedyHillClimbing()
        arcs = ArcOperatorSet()

        start = time.time()
        bn = hc.estimate(arcs, vl, start_model, patience=0)
        end = time.time()

        executions[i] = end - start

    large_results['KDEBN_' + str(n)] = pd.Series(executions, name="KDEBN_" + str(n))
    print("Large " + str(n) + " -- Time: " + str(executions.mean()) + ", std: " + str(np.std(executions, ddof=1)))

small_results.to_csv("HC_KDEBN_small.csv", index=False)
medium_results.to_csv("HC_KDEBN_medium.csv", index=False)
large_results.to_csv("HC_KDEBN_large.csv", index=False)