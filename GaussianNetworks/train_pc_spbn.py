import pandas as pd
from pybnesian import load
from pybnesian.models import SemiparametricBN
from pybnesian.learning.algorithms import GreedyHillClimbing
from pybnesian.learning.algorithms.callbacks import SaveModel
from pybnesian.learning.operators import OperatorPool, ArcOperatorSet, ChangeNodeTypeSet
from pybnesian.learning.scores import ValidatedLikelihood
import pathlib
import os
import experiments_helper


hc = GreedyHillClimbing()
change_node = ChangeNodeTypeSet()

for d in experiments_helper.DATASETS:
    for i in experiments_helper.INSTANCES:
        df = pd.read_csv(d + "_"  + str(i) + '.csv')

        pdag_lc = load('models/' + d + '/' + str(i) + '/PC/graph-lc.pickle')

        try:
            dag_lc = pdag_lc.to_dag()
        except ValueError:
            dag_lc = experiments_helper.remove_bidirected(pdag_lc)


        vl = ValidatedLikelihood(df, k=10, seed=experiments_helper.SEED)

        for p in experiments_helper.PATIENCE:
            result_folder = 'models/' + d + '/' + str(i) + '/PC/SPBN/LinearCorrelation/' + str(p)
            pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

            if not os.path.exists(result_folder + '/end.lock'):
                cb_save = SaveModel(result_folder)
                start_model = SemiparametricBN(list(df.columns.values))
                bn = hc.estimate(change_node, vl, start_model, callback=cb_save, patience=p)
                with open(result_folder + '/end.lock', 'w') as f:
                    pass


        pdag_kmi = load('models/' + d + '/' + str(i) + '/PC/graph-kmi.pickle')

        try:
            dag_kmi = pdag_kmi.to_dag()
        except ValueError:
            dag_kmi = experiments_helper.remove_bidirected(pdag_kmi)

        vl = ValidatedLikelihood(df, k=10, seed=experiments_helper.SEED)

        for p in experiments_helper.PATIENCE:
            result_folder = 'models/' + d + '/' + str(i) + '/PC/SPBN/KMutualInformation/' + str(p)
            pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

            if not os.path.exists(result_folder + '/end.lock'):
                cb_save = SaveModel(result_folder)
                start_model = SemiparametricBN(list(df.columns.values))
                bn = hc.estimate(change_node, vl, start_model, callback=cb_save, patience=p)
                with open(result_folder + '/end.lock', 'w') as f:
                    pass