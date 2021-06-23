import os
import numpy as np
np.random.seed(0)
import pandas as pd
import pathlib
import glob
from pybnesian import load
from pybnesian.factors import NodeType
from pybnesian.learning.algorithms import GreedyHillClimbing
from pybnesian.learning.algorithms.callbacks import SaveModel
from pybnesian.learning.operators import ChangeNodeTypeSet
from pybnesian.learning.scores import ValidatedLikelihood
from pybnesian.models import SemiparametricBN
import experiments_helper
import math
import multiprocessing as mp

patience = experiments_helper.PATIENCE

def find_node_types(df, dag, model_folder, type_of_dag_string, patience):
    vl = ValidatedLikelihood(df, k=10, seed=experiments_helper.SEED)

    hc = GreedyHillClimbing()
    change_node_type = ChangeNodeTypeSet()
    
    for p in patience:
        result_folder = model_folder + '/SPBN_CKDE/' + type_of_dag_string + '/' + str(p)
        pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

        if os.path.exists(result_folder + '/end.lock'):
            continue

        cb_save = SaveModel(result_folder)
        node_types = [(name, NodeType.CKDE) for name in df.columns.values]
        start_model = SemiparametricBN(dag, node_types)
        bn = hc.estimate(change_node_type, vl, start_model, callback=cb_save, patience=p)
        
        iters = sorted(glob.glob(result_folder + '/*.pickle'))
        last_file = os.path.basename(iters[-1])
        number = int(os.path.splitext(last_file)[0])
        bn.save(result_folder + '/' + str(number+1).zfill(6) + ".pickle")

        with open(result_folder + '/end.lock', 'w') as f:
            pass

def run_pc_spbn(idx_dataset, i):
    print("Folder models/" + str(idx_dataset).zfill(3) + '/' + str(i) + '/PC')

    model_folder = 'models/' + str(idx_dataset).zfill(3) + '/' + str(i) + '/PC'

    df = pd.read_csv('data/synthetic_' + str(idx_dataset).zfill(3) + '_' + str(i) + '.csv')

    pdag_lc = load(model_folder + '/graph-lc.pickle')
    try:
        dag_lc = pdag_lc.to_dag()
    except ValueError:
        dag_lc = pdag_lc.to_approximate_dag()
    find_node_types(df, dag_lc, model_folder, 'LinearCorrelation', patience)

    pdag_rcot = load(model_folder + '/graph-rcot.pickle')
    try:
        dag_rcot = pdag_rcot.to_dag()
    except ValueError:
        dag_rcot = pdag_rcot.to_approximate_dag()
    find_node_types(df, dag_rcot, model_folder, 'RCoT', patience)


for i in experiments_helper.INSTANCES:
    for idx_dataset in range(0, math.ceil(experiments_helper.NUM_SIMULATIONS / 10)):

        num_processes = min(10, experiments_helper.NUM_SIMULATIONS - idx_dataset*10)
        with mp.Pool(processes=num_processes) as p:
            p.starmap(run_pc_spbn, [(10*idx_dataset + ii, i) for ii in range(num_processes)]
                        )
