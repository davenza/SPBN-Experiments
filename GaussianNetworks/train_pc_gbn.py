import pandas as pd
from pybnesian import load
from pybnesian.models import GaussianNetwork
import pathlib
import os
import experiments_helper

for d in experiments_helper.DATASETS:
    for i in experiments_helper.INSTANCES:
        df = pd.read_csv(d + "_"  + str(i) + '.csv')

        pdag_lc = load('models/' + d + '/' + str(i) + '/PC/graph-lc.pickle')

        try:
            dag_lc = pdag_lc.to_dag()
        except ValueError:
            dag_lc = pdag_lc.to_approximate_dag()

        result_folder = 'models/' + d + '/' + str(i) + '/PC/GBN/LinearCorrelation'
        pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

        gbn_lc = GaussianNetwork(dag_lc)
        gbn_lc.save(result_folder + '/000000')

        pdag_rcot = load('models/' + d + '/' + str(i) + '/PC/graph-rcot.pickle')
        try:
            dag_rcot = pdag_rcot.to_dag()
        except ValueError:
            dag_rcot = pdag_rcot.to_approximate_dag()
        
        result_folder = 'models/' + d + '/' + str(i) + '/PC/GBN/RCoT'
        pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
        
        gbn_rcot = GaussianNetwork(dag_rcot)
        gbn_rcot.save(result_folder + '/000000')