import pandas as pd
import experiments_helper
import pathlib
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
    
    vl = ValidatedLikelihood(train_data, k=10, seed=0)
    for p in patience:
        result_folder = model_folder + '/HillClimbing/' + str(p)
        pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

        if os.path.exists(result_folder + '/end.lock'):
            continue

        cb_save = SaveModel(result_folder)
        start_model = SemiparametricBN(list(df.columns.values))
        bn = hc.estimate(pool, vl, start_model, callback=cb_save, patience=p)
        with open(result_folder + '/end.lock', 'w') as f:
            pass
