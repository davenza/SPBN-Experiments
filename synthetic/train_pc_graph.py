import pandas as pd
import experiments_helper
import pathlib
from pybnesian.learning.algorithms import PC
from pybnesian.learning.independences import LinearCorrelation, RCoT

df_200 = pd.read_csv('synthetic_200.csv')
df_2000 = pd.read_csv('synthetic_2000.csv')
df_10000 = pd.read_csv('synthetic_10000.csv')

patience = experiments_helper.PATIENCE

pc = PC()

for df, model_folder in [(df_200, 'models/200'), (df_2000, 'models/2000'), (df_10000, 'models/10000')]:
    result_folder = model_folder + '/PC'

    if not os.path.exists(result_folder + '/end-lc.lock'):
        lc = LinearCorrelation(df)
        
        graph_lc = pc.estimate(lc)
        graph_lc.save(result_folder + '/graph-lc')

        with open(result_folder + '/end-lc.lock', 'w') as f:
            pass
    
    if not os.path.exists(result_folder + '/end-rcot.lock'):
        rcot = RCoT(train_data)
        
        graph_rcot = pc.estimate(rcot)
        graph_rcot.save(result_folder + '/graph-rcot')

        with open(result_folder + '/end-rcot.lock', 'w') as f:
            pass
