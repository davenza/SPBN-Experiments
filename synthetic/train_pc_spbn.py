import numpy as np
np.random.seed(0)
import pandas as pd
import pathlib
import glob
from pybnesian import load
from pybnesian.learning.algorithms import GreedyHillClimbing
from pybnesian.learning.algorithms.callbacks import SaveModel
from pybnesian.learning.operators import ChangeNodeTypeSet
from pybnesian.learning.scores import ValidatedLikelihood
from pybnesian.models import SemiparametricBN
import experiments_helper

def find_node_types(df, dag, model_folder, type_of_dag_string, patience):
    vl = ValidatedLikelihood(df, k=10, seed=experiments_helper.SEED)

    hc = GreedyHillClimbing()
    change_node_type = ChangeNodeTypeSet()
    
    for p in patience:
        result_folder = model_folder + '/PC/' + type_of_dag_string + '/' + str(p)
        pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

        if os.path.exists(result_folder + '/end.lock'):
            continue

        cb_save = SaveModel(result_folder)
        start_model = SemiparametricBN(dag)
        bn = hc.estimate(change_node_type, vl, start_model, callback=cb_save, patience=p)


true_model = load('true_model.pickle')

df_200 = pd.read_csv('synthetic_200.csv')
df_2000 = pd.read_csv('synthetic_2000.csv')
df_10000 = pd.read_csv('synthetic_10000.csv')

patience = experiments_helper.PATIENCE

for df, model_folder in [(df_200, 'models/200'), (df_2000, 'models/2000'), (df_10000, 'models/10000')]:
    print("Folder " + model_folder)

    pdag_lc = load(model_folder + '/PC/graph-lc.pickle')
    try:
        dag_lc = pdag_lc.to_dag()
    except ValueError:
        dag_lc = experiments_helper.remove_bidirected(pdag_lc)
    find_node_types(df, dag_lc, model_folder, 'LinearCorrelation', patience)

    pdag_rcot = load(model_folder + '/PC/graph-rcot.pickle')
    try:
        dag_rcot = pdag_rcot.to_dag()
    except ValueError:
        dag_rcot = experiments_helper.remove_bidirected(pdag_rcot)
    find_node_types(df, dag_rcot, model_folder, 'RCoT', patience)
