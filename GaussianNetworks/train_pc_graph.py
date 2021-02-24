import pandas as pd
from pybnesian.learning.algorithms import PC
from pybnesian.learning.independences import LinearCorrelation, RCoT
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
    
        if not os.path.exists(result_folder + '/end-rcot.lock'):
            rcot = RCoT(df)
            
            graph_rcot = pc.estimate(rcot)
            graph_rcot.save(result_folder + '/graph-rcot')

            with open(result_folder + '/end-rcot.lock', 'w') as f:
                pass