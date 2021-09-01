import os
import glob
import numpy as np
import pandas as pd
import experiments_helper
import pathlib
from pybnesian.learning.algorithms import GreedyHillClimbing
from pybnesian.learning.algorithms.callbacks import SaveModel
from pybnesian.learning.operators import OperatorPool, ArcOperatorSet, ChangeNodeTypeSet
from pybnesian.learning.scores import ValidatedLikelihood
from pybnesian.models import SemiparametricBN
import time

patience = experiments_helper.PATIENCE

small_results = pd.DataFrame()
medium_results = pd.DataFrame()
large_results = pd.DataFrame()

for n in experiments_helper.INSTANCES:
    df = pd.read_csv('data/small_' + str(n) + ".csv")

    executions = np.empty((800,))
    for i in range(800):
        if i % 10 == 0:
            print(str(i) + " executions")
        vl = ValidatedLikelihood(df, k=10, seed=i)
        start_model = SemiparametricBN(list(df.columns.values))
        hc = GreedyHillClimbing()
        pool = OperatorPool([ArcOperatorSet(), ChangeNodeTypeSet()])

        start = time.time()
        bn = hc.estimate(pool, vl, start_model, patience=0)
        end = time.time()

        executions[i] = end - start

    small_results['SPBN_' + str(n)] = pd.Series(executions, name="SPBN_" + str(n))
    print("Small " + str(n) + " -- Time: " + str(executions.mean()) + ", std: " + str(np.std(executions, ddof=1)))

    df = pd.read_csv('data/medium_' + str(n) + ".csv")

    executions = np.empty((200,))
    for i in range(200):
        print(str(i) + " executions")
        vl = ValidatedLikelihood(df, k=10, seed=i)
        start_model = SemiparametricBN(list(df.columns.values))
        hc = GreedyHillClimbing()
        pool = OperatorPool([ArcOperatorSet(), ChangeNodeTypeSet()])

        start = time.time()
        bn = hc.estimate(pool, vl, start_model, patience=0)
        end = time.time()

        executions[i] = end - start

    medium_results['SPBN_' + str(n)] = pd.Series(executions, name="SPBN_" + str(n))
    print("Medium " + str(n) + " -- Time: " + str(executions.mean()) + ", std: " + str(np.std(executions, ddof=1)))

    df = pd.read_csv('data/large_' + str(n) + ".csv")

    executions = np.empty((80,))
    for i in range(80):
        print(str(i) + " executions")
        vl = ValidatedLikelihood(df, k=10, seed=i)
        start_model = SemiparametricBN(list(df.columns.values))
        hc = GreedyHillClimbing()
        pool = OperatorPool([ArcOperatorSet(), ChangeNodeTypeSet()])

        start = time.time()
        bn = hc.estimate(pool, vl, start_model, patience=0)
        end = time.time()

        executions[i] = end - start

    large_results['SPBN_' + str(n)] = pd.Series(executions, name="SPBN_" + str(n))
    print("Large " + str(n) + " -- Time: " + str(executions.mean()) + ", std: " + str(np.std(executions, ddof=1)))

if os.path.exists('HC_SPBN_small.csv'):
    current_small = pd.read_csv("HC_SPBN_small.csv")
    small_results = pd.concat([current_small, small_results])

small_results.to_csv("HC_SPBN_small.csv", index=False)

if os.path.exists('HC_SPBN_medium.csv'):
    current_medium = pd.read_csv("HC_SPBN_medium.csv")
    medium_results = pd.concat([current_medium, medium_results])

medium_results.to_csv("HC_SPBN_medium.csv", index=False)

if os.path.exists('HC_SPBN_large.csv'):
    current_large = pd.read_csv('HC_SPBN_large.csv')
    large_results = pd.concat([current_large, large_results])

large_results.to_csv("HC_SPBN_large.csv", index=False)