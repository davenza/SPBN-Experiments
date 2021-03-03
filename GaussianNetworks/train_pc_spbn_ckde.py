import glob
import pandas as pd
from pybnesian import load
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
change_node = ChangeNodeTypeSet()

for d in experiments_helper.DATASETS:
    for i in experiments_helper.INSTANCES:
        df = pd.read_csv(d + "_"  + str(i) + '.csv')

        pdag_lc = load('models/' + d + '/' + str(i) + '/PC/graph-lc.pickle')

        try:
            dag_lc = pdag_lc.to_dag()
        except ValueError:
            dag_lc = pdag_lc.to_approximate_dag()


        vl = ValidatedLikelihood(df, k=10, seed=experiments_helper.SEED)

        for p in experiments_helper.PATIENCE:
            result_folder = 'models/' + d + '/' + str(i) + '/PC/SPBN_CKDE/LinearCorrelation/' + str(p)
            pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

            if not os.path.exists(result_folder + '/end.lock'):
                cb_save = SaveModel(result_folder)
                node_types = [(name, NodeType.CKDE) for name in df.columns.values]
                start_model = SemiparametricBN(dag_lc, node_types)
                bn = hc.estimate(change_node, vl, start_model, callback=cb_save, patience=p)

                iters = sorted(glob.glob(result_folder + '/*.pickle'))
                last_file = os.path.basename(iters[-1])
                number = int(os.path.splitext(last_file)[0])
                bn.save(result_folder + '/' + str(number+1).zfill(6) + ".pickle")

                with open(result_folder + '/end.lock', 'w') as f:
                    pass


        pdag_rcot = load('models/' + d + '/' + str(i) + '/PC/graph-rcot.pickle')

        try:
            dag_rcot = pdag_rcot.to_dag()
        except ValueError:
            dag_rcot = pdag_rcot.to_approximate_dag()

        vl = ValidatedLikelihood(df, k=10, seed=experiments_helper.SEED)

        for p in experiments_helper.PATIENCE:
            result_folder = 'models/' + d + '/' + str(i) + '/PC/SPBN_CKDE/RCoT/' + str(p)
            pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

            if not os.path.exists(result_folder + '/end.lock'):
                cb_save = SaveModel(result_folder)
                node_types = [(name, NodeType.CKDE) for name in df.columns.values]
                start_model = SemiparametricBN(dag_rcot, node_types)
                bn = hc.estimate(change_node, vl, start_model, callback=cb_save, patience=p)

                iters = sorted(glob.glob(result_folder + '/*.pickle'))
                last_file = os.path.basename(iters[-1])
                number = int(os.path.splitext(last_file)[0])
                bn.save(result_folder + '/' + str(number+1).zfill(6) + ".pickle")

                with open(result_folder + '/end.lock', 'w') as f:
                    pass