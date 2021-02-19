import pandas as pd
import experiments_helper
import pathlib
from pybnesian.learning.algorithms import PC
from pybnesian.learning.independences import LinearCorrelation, KMutualInformation

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
    
    if not os.path.exists(result_folder + '/end-kmi.lock'):
        kmi = KMutualInformation(train_data, k=25, seed=experiments_helper.SEED)
        
        graph_kmi = pc.estimate(kmi)
        graph_kmi.save(result_folder + '/graph-kmi')

        with open(result_folder + '/end-kmi.lock', 'w') as f:
            pass
