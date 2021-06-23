import glob
import pandas as pd
from pybnesian.models import GaussianNetwork
from pybnesian.learning.algorithms import GreedyHillClimbing
from pybnesian.learning.algorithms.callbacks import SaveModel
from pybnesian.learning.operators import OperatorPool, ArcOperatorSet, ChangeNodeTypeSet
from pybnesian.learning.scores import ValidatedLikelihood, BIC, BGe
import pathlib
import os
import experiments_helper



def train_gbn(dataset, idx_dataset, instances):
    df = pd.read_csv('data/' + dataset + "_" + str(idx_dataset).zfill(3) + "_" + str(instances) + '.csv')

    hc = GreedyHillClimbing()
    arc_set = ArcOperatorSet()
    result_folder = 'models/' + dataset + '/' + str(idx_dataset).zfill(3) + '/' + str(instances) + '/HillClimbing/GBN_BIC/'
    pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(result_folder + '/end.lock'):
        bic = BIC(df)
        cb_save = SaveModel(result_folder)
        start_model = GaussianNetwork(list(df.columns.values))
        bn = hc.estimate(arc_set, bic, start_model, callback=cb_save, verbose=True)

        iters = sorted(glob.glob(result_folder + '/*.pickle'))
        last_file = os.path.basename(iters[-1])
        number = int(os.path.splitext(last_file)[0])
        bn.save(result_folder + '/' + str(number+1).zfill(6) + ".pickle")

        with open(result_folder + '/end.lock', 'w') as f:
            pass

    hc = GreedyHillClimbing()
    arc_set = ArcOperatorSet()
    result_folder = 'models/' + dataset + '/' + str(idx_dataset).zfill(3) + '/' + str(instances) + '/HillClimbing/GBN_BGe/'
    pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(result_folder + '/end.lock'):
        bge = BGe(df)
        cb_save = SaveModel(result_folder)
        start_model = GaussianNetwork(list(df.columns.values))
        bn = hc.estimate(arc_set, bge, start_model, callback=cb_save, verbose=True)

        iters = sorted(glob.glob(result_folder + '/*.pickle'))
        last_file = os.path.basename(iters[-1])
        number = int(os.path.splitext(last_file)[0])
        bn.save(result_folder + '/' + str(number+1).zfill(6) + ".pickle")

        with open(result_folder + '/end.lock', 'w') as f:
            pass

for d in experiments_helper.DATASETS:
    for idx_dataset in range(experiments_helper.NUM_SIMULATIONS):
        for i in experiments_helper.INSTANCES:
            train_gbn(d, idx_dataset, i)