import pandas as pd
from pgmpy.models import HybridContinuousModel
from pgmpy.estimators import HybridCachedHillClimbing, ValidationLikelihood, GaussianValidationLikelihood, \
                            GaussianBicScore, BGeScore, CachedHillClimbing
from pgmpy.estimators.callbacks import DrawModel, SaveModel

import pathlib
import os

datasets = ["ecoli70", "magic_niab", "magic_irri", "arth150"]
instances = [200, 2000, 10000]
patience = [0, 5]


for d in datasets:
    for i in instances:
        df = pd.read_csv(d + "_"  + str(i) + '.csv')

        for p in patience:
            result_folder = 'models/' + d + '/' + str(i) + '/CKDE/' + str(p)
            pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

            if not os.path.exists(result_folder + '/end.lock'):
                vl = ValidationLikelihood(df, k=10, seed=0)
                hc = HybridCachedHillClimbing(df, scoring_method=vl)
                cb_draw = DrawModel(result_folder)
                cb_save = SaveModel(result_folder)
                bn = hc.estimate(callbacks=[cb_draw, cb_save], patience=p)
                with open(result_folder + '/end.lock', 'w') as f:
                    pass

        for p in patience:
            result_folder = 'models/' + d + '/' + str(i) + '/GBN_Validation/' + str(p)
            pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

            if not os.path.exists(result_folder + '/end.lock'):
                gv = GaussianValidationLikelihood(df, k=10, seed=0)
                ghc = CachedHillClimbing(df, scoring_method=gv)
                cb_draw = DrawModel(result_folder)
                cb_save = SaveModel(result_folder)
                gbn = ghc.estimate(callbacks=[cb_draw, cb_save], patience=p)
                with open(result_folder + '/end.lock', 'w') as f:
                    pass

        result_folder = 'models/' + d + '/' + str(i) + '/GBN_BIC/'
        if not os.path.exists(result_folder + '/end.lock'):
            pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
            bic = GaussianBicScore(df)
            ghc = CachedHillClimbing(df, scoring_method=bic)
            cb_draw = DrawModel(result_folder)
            cb_save = SaveModel(result_folder)
            gbn = ghc.estimate(callbacks=[cb_draw, cb_save])
            with open(result_folder + '/end.lock', 'w') as f:
                pass

        result_folder = 'models/' + d + '/' + str(i) + '/GBN_BGe/'
        if not os.path.exists(result_folder + '/end.lock'):
            pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)
            bge = BGeScore(df)
            ghc = CachedHillClimbing(df, scoring_method=bge)
            cb_draw = DrawModel(result_folder)
            cb_save = SaveModel(result_folder)
            gbn = ghc.estimate(callbacks=[cb_draw, cb_save])
            with open(result_folder + '/end.lock', 'w') as f:
                pass