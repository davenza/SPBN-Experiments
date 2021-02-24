import pandas as pd
from pybnesian import load
from pybnesian.models import GaussianNetwork
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

        gbn_lc = GaussianNetwork(dag_lc)
        gbn_lc.save('models/' + d + '/' + str(i) + '/PC/GBN/LinearCorrelation/000000')

        pdag_rcot = load('models/' + d + '/' + str(i) + '/PC/graph-rcot.pickle')

        try:
            dag_rcot = pdag_rcot.to_dag()
        except ValueError:
            dag_rcot = experiments_helper.remove_bidirected(pdag_rcot)

        gbn_rcot = GaussianNetwork(dag_rcot)
        gbn_rcot.save('models/' + d + '/' + str(i) + '/PC/GBN/RCoT/000000')