import numpy as np
np.random.seed(0)
import pandas as pd
import pathlib
import glob
from pybnesian import load
import experiments_helper

true_model = load('true_model.pickle')

df_200 = pd.read_csv('synthetic_200.csv')
df_2000 = pd.read_csv('synthetic_2000.csv')
df_10000 = pd.read_csv('synthetic_10000.csv')
df_test = pd.read_csv('synthetic_test.csv')

print("True model logl: " + str(true_model.slogl(df_test)))

patience = experiments_helper.PATIENCE

for df, model_folder in [(df_200, 'models/200'), (df_2000, 'models/2000'), (df_10000, 'models/10000')]:
    print("Folder " + model_folder)
    for p in patience:
        result_folder = model_folder + '/HillClimbing/SPBN/' + str(p)
        pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

        all_models = sorted(glob.glob(result_folder + '/*.pickle'))
        final_model = load(all_models[-1])
        final_model.fit(df)

        slogl = final_model.slogl(df_test)

        print("Loglik, p " + str(p) + ": " + str(slogl))
        print("SHD, p " + str(p) + ": " + str(experiments_helper.shd(final_model, true_model)))
        print("Hamming, p " + str(p) + ": " + str(experiments_helper.hamming(final_model, true_model)))
        print("Hamming type, p " + str(p) + ": " + str(experiments_helper.hamming_type(final_model, true_model)))

        print()



