import glob
import pandas as pd
from pybnesian.factors import NodeType
from pybnesian.models import SemiparametricBN
from pybnesian.learning.algorithms import GreedyHillClimbing
from pybnesian.learning.algorithms.callbacks import SaveModel
from pybnesian.learning.operators import OperatorPool, ArcOperatorSet, ChangeNodeTypeSet
from pybnesian.learning.scores import ValidatedLikelihood
import pathlib
import os
import experiments_helper

hc = GreedyHillClimbing()
pool = OperatorPool([ArcOperatorSet(), ChangeNodeTypeSet()])

for d in experiments_helper.DATASETS:
    for i in experiments_helper.INSTANCES:
        for idx_dataset in range(experiments_helper.NUM_SIMULATIONS):
            df = pd.read_csv('data/' + d + "_" + str(idx_dataset).zfill(3) + "_" + str(i) + '.csv')


            vl = ValidatedLikelihood(df, k=10, seed=experiments_helper.SEED)

            for p in experiments_helper.PATIENCE:
                result_folder = 'models/' + d + '/' + str(idx_dataset).zfill(3) + '/' + str(i) + '/HillClimbing/SPBN_CKDE/' + str(p)
                pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

                if not os.path.exists(result_folder + '/end.lock'):
                    cb_save = SaveModel(result_folder)
                    node_types = [(name, NodeType.CKDE) for name in df.columns.values]
                    start_model = SemiparametricBN(list(df.columns.values), node_types)
                    bn = hc.estimate(pool, vl, start_model, callback=cb_save, patience=p, verbose=True)

                    iters = sorted(glob.glob(result_folder + '/*.pickle'))
                    last_file = os.path.basename(iters[-1])
                    number = int(os.path.splitext(last_file)[0])
                    bn.save(result_folder + '/' + str(number+1).zfill(6) + ".pickle")
                    
                    with open(result_folder + '/end.lock', 'w') as f:
                        pass