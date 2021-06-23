import pandas as pd
from pybnesian import load
from pybnesian.models import GaussianNetwork
import pathlib
import os
import experiments_helper

for d in experiments_helper.DATASETS:
    for i in experiments_helper.INSTANCES:
        for idx_dataset in range(experiments_helper.NUM_SIMULATIONS):
            df = pd.read_csv('data/' + d + "_" + str(idx_dataset).zfill(3) + "_" + str(i) + '.csv')

            pdag_lc = load('models/' + d + '/' + str(idx_dataset).zfill(3) + '/' + str(i) + '/PC/graph-lc.pickle')

            try:
                dag_lc = pdag_lc.to_dag()
            except ValueError:
                dag_lc = pdag_lc.to_approximate_dag()

            result_folder = 'models/' + d + '/' + str(idx_dataset).zfill(3) + '/' + str(i) + '/PC/GBN/LinearCorrelation'
            pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

            gbn_lc = GaussianNetwork(dag_lc)
            gbn_lc.save(result_folder + '/000000')

            pdag_rcot = load('models/' + d + '/' + str(idx_dataset).zfill(3) + '/' + str(i) + '/PC/graph-rcot.pickle')
            try:
                dag_rcot = pdag_rcot.to_dag()
            except ValueError:
                dag_rcot = pdag_rcot.to_approximate_dag()
            
            result_folder = 'models/' + d + '/' + str(idx_dataset).zfill(3) + '/' + str(i) + '/PC/GBN/RCoT'
            pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
            
            gbn_rcot = GaussianNetwork(dag_rcot)
            gbn_rcot.save(result_folder + '/000000')