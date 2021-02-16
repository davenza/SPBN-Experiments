import glob
import os
import numpy as np
np.random.seed(0)
import pandas as pd
from pgmpy.estimators import HybridCachedHillClimbingStrict, ValidationSPBNStrict
from pgmpy.estimators.callbacks import DrawModel, SaveModel
from pgmpy.models import HybridContinuousModel

from sklearn.model_selection import KFold
import pathlib

import multiprocessing as mp
import experiments_helper


def run_validation_spbn_strict(train_data, folds, patience, result_folder, idx_fold):
    for k in folds:
        for p in patience:
            fold_folder = result_folder + '/SPBN_Strict/Validation_' + str(k) + '_' + str(p) + '/' + str(idx_fold)
            pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

            if os.path.exists(fold_folder + '/end.lock'):
                continue

            vl = ValidationSPBNStrict(train_data, k=k, seed=0)
            hc = HybridCachedHillClimbingStrict(train_data, scoring_method=vl)
            cb_draw = DrawModel(fold_folder)
            cb_save = SaveModel(fold_folder)
            bn = hc.estimate(callbacks=[cb_draw, cb_save], patience=p)
            with open(fold_folder + '/end.lock', 'w') as f:
                pass


def train_crossvalidation_file(file, folds, patience):
    x = experiments_helper.validate_dataset(file, [2, 3, 5, 10])
    if x is None:
        return
    else:
        dataset, result_folder = x

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    if not os.path.exists(result_folder + '/SPBN_Strict'):
        os.mkdir(result_folder + '/SPBN_Strict')

    with mp.Pool(processes=10) as p:
        p.starmap(run_validation_spbn_strict, [(dataset.iloc[train_indices,:], folds, patience, result_folder, idx_fold)
                                             for (idx_fold, (train_indices, test_indices)) in
                                             enumerate(KFold(10, shuffle=True, random_state=0).split(dataset))]
                  )


def train_crossvalidation():
    files = experiments_helper.find_crossvalidation_datasets()

    folds = [10]
    patience = [0, 5]

    for file in files:
        train_crossvalidation_file(file, folds, patience)


def test_validation_spbn_strict(train_data, test_data, folds, patience, result_folder, idx_fold):
    test_scores = np.full((len(folds), len(patience)), np.nan)

    for idx_k, k in enumerate(folds):
        for idx_p, p in enumerate(patience):
            models_folder = result_folder + '/SPBN_Strict/Validation_' + str(k) + "_" + str(p) + '/' + str(idx_fold)
            all_models = sorted(glob.glob(models_folder + '/*.pkl'))
            final_model = HybridContinuousModel.load_model(all_models[-1])
            final_model.fit(train_data)
            test_scores[idx_k, idx_p] = final_model.logpdf_dataset(test_data).sum()

    return test_scores


def test_crossvalidation():
    files = experiments_helper.find_crossvalidation_datasets()

    folds = [10]
    patience = [0, 5]

    string_file = "Dataset," + ','.join(["SPBN_STRICT_"+ str(f) + "_" + str(p) for f in folds for p in patience])

    print(string_file)
    for file in files:
        x = experiments_helper.validate_dataset(file, [2, 3, 5, 10])
        if x is None:
            continue
        else:
            dataset, result_folder = x

        validation_spbn_strict = np.full((10, len(folds), len(patience)), np.nan)

        for (idx_fold, (train_indices, test_indices)) in enumerate(KFold(10, shuffle=True, random_state=0).split(dataset)):
            train_dataset = dataset.iloc[train_indices,:]
            test_dataset = dataset.iloc[test_indices,:]

            validation_spbn_strict[idx_fold] = test_validation_spbn_strict(train_dataset, test_dataset, folds, patience,
                                                             result_folder, idx_fold)

        sum_validation_spbn_strict = validation_spbn_strict.sum(axis=0)

        basefolder = os.path.basename(os.path.dirname(file))
        new_line = basefolder

        for idx_f, f in enumerate(folds):
            for idx_p, p in enumerate(patience):
                new_line += "," + str(sum_validation_spbn_strict[idx_f, idx_p])
        print(new_line)

        string_file += '\n' + new_line

    with open('cv_results_spbn_strict.csv', 'w') as f:
        f.write(string_file)

def run_holdout():
    pass



if __name__ == '__main__':
    # train_crossvalidation()
    # train_crossvalidation_file('data/MFeatures/mfeat.csv', [2, 3, 5, 10], [0, 5])
    test_crossvalidation()
