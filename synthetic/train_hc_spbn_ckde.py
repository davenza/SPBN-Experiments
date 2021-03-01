import os
import glob
import pandas as pd
import experiments_helper
import pathlib
from pybnesian.factors import NodeType
from pybnesian.learning.algorithms import GreedyHillClimbing
from pybnesian.learning.algorithms.callbacks import SaveModel
from pybnesian.learning.operators import OperatorPool, ArcOperatorSet, ChangeNodeTypeSet
from pybnesian.learning.scores import ValidatedLikelihood
from pybnesian.models import SemiparametricBN

df_200 = pd.read_csv('synthetic_200.csv')
df_2000 = pd.read_csv('synthetic_2000.csv')
df_10000 = pd.read_csv('synthetic_10000.csv')

patience = experiments_helper.PATIENCE

hc = GreedyHillClimbing()
pool = OperatorPool([ArcOperatorSet(), ChangeNodeTypeSet()])

for df, model_folder in [(df_200, 'models/200'), (df_2000, 'models/2000'), (df_10000, 'models/10000')]:
    
    vl = ValidatedLikelihood(df, k=10, seed=0)
    for p in patience:
        result_folder = model_folder + '/HillClimbing/SPBN_CKDE/' + str(p)
        pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

        if os.path.exists(result_folder + '/end.lock'):
            continue

        cb_save = SaveModel(result_folder)
        node_types = [(name, NodeType.CKDE) for name in df.columns.values]
        start_model = SemiparametricBN(list(df.columns.values), node_types)
        bn = hc.estimate(pool, vl, start_model, callback=cb_save, patience=p)
        
        iters = sorted(glob.glob(result_folder + '/*.pickle'))
        last_file = os.path.basename(iters[-1])
        number = int(os.path.splitext(last_file)[0])
        bn.save(result_folder + '/' + str(number+1).zfill(6) + ".pickle")
        with open(result_folder + '/end.lock', 'w') as f:
            pass
