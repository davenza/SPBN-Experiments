import os
import numpy as np
import pandas as pd
import experiments_helper
import pathlib
from pybnesian.learning.algorithms import PC, GreedyHillClimbing
from pybnesian.learning.independences import LinearCorrelation, RCoT
from pybnesian.learning.operators import ChangeNodeTypeSet
from pybnesian.learning.scores import ValidatedLikelihood
from pybnesian.models import SemiparametricBN
import time


patience = experiments_helper.PATIENCE


def test_pc_time(df, n_exec):
    local_results = pd.DataFrame()

    pc = PC()
    lc = LinearCorrelation(df)
    
    executions = np.empty((n_exec,))
    for i in range(n_exec):
        start = time.time()
        graph_lc = pc.estimate(lc)
        end = time.time()

        executions[i] = end - start

    local_results['PC-LC-Graph'] = pd.Series(executions, name="PC-LC-Graph")
    print("LC Graph Time: " + str(executions.mean()))

    try:
        dag = graph_lc.to_dag()
    except ValueError:
        dag = graph_lc.to_approximate_dag()

    executions = np.empty((n_exec,))
    for i in range(n_exec):
        hc = GreedyHillClimbing()
        change_node_type = ChangeNodeTypeSet()
        vl = ValidatedLikelihood(df, k=10, seed=experiments_helper.SEED)

        start_model = SemiparametricBN(dag)

        start = time.time()
        bn = hc.estimate(change_node_type, vl, start_model, patience=0)
        end = time.time()

        executions[i] = end - start
    
    local_results['PC-LC-NodeType'] = pd.Series(executions, name="PC-LC-NodeType")
    print("LC HC NodeType Time: " + str(executions.mean()))
    
    rcot = RCoT(df)
    
    for i in range(n_exec):
        start = time.time()
        graph_rcot = pc.estimate(rcot) 
        end = time.time()

        executions[i] = end - start

    local_results['PC-RCoT-Graph'] = pd.Series(executions, name="PC-RCoT-Graph")
    print("RCoT Graph Time: " + str(executions.mean()))

    for i in range(n_exec):
        hc = GreedyHillClimbing()
        change_node_type = ChangeNodeTypeSet()
        vl = ValidatedLikelihood(df, k=10, seed=experiments_helper.SEED)

        try:
            dag = graph_rcot.to_dag()
        except ValueError:
            dag = graph_rcot.to_approximate_dag()

        start_model = SemiparametricBN(dag)
        
        start = time.time()
        bn = hc.estimate(change_node_type, vl, start_model, patience=0)
        end = time.time()

        executions[i] = end - start

    local_results['PC-RCoT-NodeType'] = pd.Series(executions, name="PC-RCoT-NodeType")
    print("RCoT HC NodeType Time: " + str(executions.mean()))

    return local_results


small_results = pd.DataFrame()
medium_results = pd.DataFrame()
large_results = pd.DataFrame()

for n in experiments_helper.INSTANCES:

    print("Small " + str(n) + " instances")
    print("=====================")

    df = pd.read_csv('data/small_' + str(n) + ".csv")
    local_results = test_pc_time(df, 200)

    small_results['Graph-LC_' + str(n)] = local_results['PC-LC-Graph']
    small_results['Graph-RCoT_' + str(n)] = local_results['PC-RCoT-Graph']
    small_results['NodeType-LC_' + str(n)] = local_results['PC-LC-NodeType']
    small_results['NodeType-RCoT_' + str(n)] = local_results['PC-RCoT-NodeType']

    print("Medium " + str(n) + " instances")
    print("=====================")

    df = pd.read_csv('data/medium_' + str(n) + ".csv")
    local_results = test_pc_time(df, 50)

    medium_results['Graph-LC_' + str(n)] = local_results['PC-LC-Graph']
    medium_results['Graph-RCoT_' + str(n)] = local_results['PC-RCoT-Graph']
    medium_results['NodeType-LC_' + str(n)] = local_results['PC-LC-NodeType']
    medium_results['NodeType-RCoT_' + str(n)] = local_results['PC-RCoT-NodeType']

    print("Large " + str(n) + " instances")
    print("=====================")

    df = pd.read_csv('data/large_' + str(n) + ".csv")
    local_results = test_pc_time(df, 20)

    large_results['Graph-LC_' + str(n)] = local_results['PC-LC-Graph']
    large_results['Graph-RCoT_' + str(n)] = local_results['PC-RCoT-Graph']
    large_results['NodeType-LC_' + str(n)] = local_results['PC-LC-NodeType']
    large_results['NodeType-RCoT_' + str(n)] = local_results['PC-RCoT-NodeType']

if os.path.exists('PC_small.csv'):
    current_small = pd.read_csv("PC_small.csv")
    small_results = pd.concat([current_small, small_results])

small_results.to_csv("PC_small.csv", index=False)

if os.path.exists('PC_medium.csv'):
    current_medium = pd.read_csv("PC_medium.csv")
    medium_results = pd.concat([current_medium, medium_results])

medium_results.to_csv("PC_medium.csv", index=False)

if os.path.exists('PC_large.csv'):
    current_large = pd.read_csv('PC_large.csv')
    large_results = pd.concat([current_large, large_results])

large_results.to_csv("PC_large.csv", index=False)