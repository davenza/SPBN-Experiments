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

        pdag_kmi = load('models/' + d + '/' + str(i) + '/PC/graph-kmi.pickle')

        try:
            dag_kmi = pdag_kmi.to_dag()
        except ValueError:
            dag_kmi = experiments_helper.remove_bidirected(pdag_kmi)

        gbn_kmi = GaussianNetwork(dag_kmi)
        gbn_kmi.save('models/' + d + '/' + str(i) + '/PC/GBN/KMutualInformation/000000')