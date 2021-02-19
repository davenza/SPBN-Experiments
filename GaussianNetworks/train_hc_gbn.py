import pandas as pd
from pybnesian.models import GaussianNetwork
from pybnesian.learning.algorithms import GreedyHillClimbing
from pybnesian.learning.algorithms.callbacks import SaveModel
from pybnesian.learning.operators import OperatorPool, ArcOperatorSet, ChangeNodeTypeSet
from pybnesian.learning.scores import ValidatedLikelihood, BIC, BGe
import pathlib
import os
import experiments_helper

hc = GreedyHillClimbing()
arc_set = ArcOperatorSet()

for d in experiments_helper.DATASETS:
    for i in experiments_helper.INSTANCES:
        df = pd.read_csv(d + "_"  + str(i) + '.csv')

        vl = ValidatedLikelihood(df, k=10, seed=experiments_helper.SEED)

        for p in experiments_helper.PATIENCE:
            result_folder = 'models/' + d + '/' + str(i) + '/HillClimbing/GBN_Validation/' + str(p)
            pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
            if not os.path.exists(result_folder + '/end.lock'):
                cb_save = SaveModel(result_folder)
                start_model = GaussianNetwork(list(df.columns.values))
                bn = hc.estimate(arc_set, vl, start_model, callback=cb_save, patience=p)
                with open(result_folder + '/end.lock', 'w') as f:
                    pass

        result_folder = 'models/' + d + '/' + str(i) + '/HillClimbing/GBN_BIC/'
        pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(result_folder + '/end.lock'):
            bic = BIC(df)
            cb_save = SaveModel(result_folder)
            start_model = GaussianNetwork(list(df.columns.values))
            bn = hc.estimate(arc_set, bic, start_model, callback=cb_save)
            with open(result_folder + '/end.lock', 'w') as f:
                pass

        result_folder = 'models/' + d + '/' + str(i) + '/HillClimbing/GBN_BGe/'
        pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(result_folder + '/end.lock'):
            bge = BGe(df)
            cb_save = SaveModel(result_folder)
            start_model = GaussianNetwork(list(df.columns.values))
            bn = hc.estimate(arc_set, bge, start_model, callback=cb_save)
            with open(result_folder + '/end.lock', 'w') as f:
                pass