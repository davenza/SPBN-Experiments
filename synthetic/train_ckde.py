import pandas as pd
from pgmpy.models import HybridContinuousModel
from pgmpy.estimators import HybridCachedHillClimbing, ValidationLikelihood
from pgmpy.estimators.callbacks import DrawModel, SaveModel

import pathlib

df_200 = pd.read_csv('synthetic_200.csv')
df_2000 = pd.read_csv('synthetic_2000.csv')
df_10000 = pd.read_csv('synthetic_10000.csv')

patience = [0, 5]

for df, model_folder in [(df_200, 'models/200'), (df_2000, 'models/2000'), (df_10000, 'models/10000')]:
    for p in patience:
        result_folder = model_folder + '/' + str(p)
        pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

        vl = ValidationLikelihood(df, k=10, seed=0)
        hc = HybridCachedHillClimbing(df, scoring_method=vl)
        cb_draw = DrawModel(result_folder)
        cb_save = SaveModel(result_folder)
        bn = hc.estimate(callbacks=[cb_draw, cb_save], patience=p)