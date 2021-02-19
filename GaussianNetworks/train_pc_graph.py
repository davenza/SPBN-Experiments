import pandas as pd
from pybnesian.learning.algorithms import PC
from pybnesian.learning.independences import LinearCorrelation, KMutualInformation
import pathlib
import os
import experiments_helper

pc = PC()

for d in experiments_helper.DATASETS:
    for i in experiments_helper.INSTANCES:
        df = pd.read_csv(d + "_"  + str(i) + '.csv')
        result_folder = 'models/' + d + '/' + str(i) + '/PC/'
        pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
    
        if not os.path.exists(result_folder + '/end-lc.lock'):
            lc = LinearCorrelation(df)
            
            graph_lc = pc.estimate(lc)
            graph_lc.save(result_folder + '/graph-lc')

            with open(result_folder + '/end-lc.lock', 'w') as f:
                pass
    
        if not os.path.exists(result_folder + '/end-kmi.lock'):
            kmi = KMutualInformation(df, k=25, seed=experiments_helper.SEED)
            
            graph_kmi = pc.estimate(kmi)
            graph_kmi.save(result_folder + '/graph-kmi')

            with open(result_folder + '/end-kmi.lock', 'w') as f:
                pass