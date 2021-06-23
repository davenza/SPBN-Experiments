import os
import glob
import pandas as pd
import experiments_helper
import pathlib
from pybnesian.learning.algorithms import GreedyHillClimbing
from pybnesian.learning.algorithms.callbacks import SaveModel
from pybnesian.learning.operators import OperatorPool, ArcOperatorSet, ChangeNodeTypeSet
from pybnesian.learning.scores import ValidatedLikelihood
from pybnesian.models import SemiparametricBN
import math
import multiprocessing as mp


patience = experiments_helper.PATIENCE

hc = GreedyHillClimbing()
pool = OperatorPool([ArcOperatorSet(), ChangeNodeTypeSet()])


def run_hc_spbn(idx_dataset, i):
    df = pd.read_csv('data/synthetic_' + str(idx_dataset).zfill(3) + '_' + str(i) + '.csv')

    vl = ValidatedLikelihood(df, k=10, seed=0)
    for p in patience:
        result_folder = 'models/' + str(idx_dataset).zfill(3) + '/' + str(i) + '/HillClimbing/SPBN/' + str(p)
        pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

        if os.path.exists(result_folder + '/end.lock'):
            continue

        cb_save = SaveModel(result_folder)
        start_model = SemiparametricBN(list(df.columns.values))
        bn = hc.estimate(pool, vl, start_model, callback=cb_save, patience=p)

        iters = sorted(glob.glob(result_folder + '/*.pickle'))
        last_file = os.path.basename(iters[-1])
        number = int(os.path.splitext(last_file)[0])
        bn.save(result_folder + '/' + str(number+1).zfill(6) + ".pickle")
        with open(result_folder + '/end.lock', 'w') as f:
            pass


for i in experiments_helper.INSTANCES:
    for idx_dataset in range(0, math.ceil(experiments_helper.NUM_SIMULATIONS / 10)):

        num_processes = min(10, experiments_helper.NUM_SIMULATIONS - idx_dataset*10)
        with mp.Pool(processes=num_processes) as p:
            p.starmap(run_hc_spbn, [(10*idx_dataset + ii, i) for ii in range(num_processes)]
                        )