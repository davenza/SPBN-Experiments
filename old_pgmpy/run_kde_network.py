import os
import experiments_helper
import multiprocessing as mp
import pathlib

import numpy as np
from sklearn.model_selection import KFold
import glob

import pickle

from pgmpy.estimators import ValidationConditionalKDE, CachedHillClimbing
from pgmpy.estimators.callbacks import DrawModel, SaveModel
from pgmpy.models import KDEBayesianNetwork

def run_kde_network(train_data, folds, patience, result_folder, idx_fold):
    for f in folds:
        for p in patience:
            fold_folder = result_folder + '/KDE_network/Validation_' + str(f) + "_" + str(p) + '/' + str(idx_fold)
            pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

            if os.path.exists(fold_folder + '/end.lock'):
                continue

            vl = ValidationConditionalKDE(train_data, k=f, seed=0)
            hc = CachedHillClimbing(train_data, scoring_method=vl)
            cb_draw = DrawModel(fold_folder)
            cb_save = SaveModel(fold_folder)

            hc.estimate(callbacks=[cb_draw, cb_save], patience=p)
            with open(fold_folder + '/end.lock', 'w') as file_lock:
                pass


def train_crossvalidation_file(file, folds, patience):
    print("Training " + str(file))

    x = experiments_helper.validate_dataset(file, [2, 3, 5, 10])
    if x is None:
        return
    else:
        dataset, result_folder = x

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    if not os.path.exists(result_folder + '/KDE_network'):
        os.mkdir(result_folder + '/KDE_network')

    for (idx_fold, (train_indices, test_indices)) in enumerate(KFold(10, shuffle=True, random_state=0).split(dataset)):
        run_kde_network(dataset.iloc[train_indices,:], folds, patience, result_folder, idx_fold)

    # with mp.Pool(processes=10) as p:
    #     p.starmap(run_kde_network, [(dataset.iloc[train_indices,:], folds, patience, result_folder, idx_fold)
    #                                          for (idx_fold, (train_indices, test_indices)) in
    #                                          enumerate(KFold(10, shuffle=True, random_state=0).split(dataset))]
    #               )

def train_crossvalidation():
    files = experiments_helper.find_crossvalidation_datasets()

    folds = [10]
    patience = [0, 5]

    for file in files:
        train_crossvalidation_file(file, folds, patience)


def test_validation_kde_network(train_data, test_data, folds, patience, result_folder, idx_fold):
    test_scores = np.full((len(folds), len(patience)), np.nan)

    for idx_k, k in enumerate(folds):
        for idx_p, p in enumerate(patience):
            models_folder = result_folder + '/KDE_network/Validation_' + str(k) + "_" + str(p) + '/' + str(idx_fold)
            all_models = sorted(glob.glob(models_folder + '/*.pkl'))
            final_model = KDEBayesianNetwork.load_model(all_models[-1])
            final_model.fit(train_data)
            test_scores[idx_k, idx_p] = final_model.logpdf_dataset(test_data).sum()

    return test_scores

def test_crossvalidation():
    files = experiments_helper.find_crossvalidation_datasets()

    folds = [10]
    patience = [0, 5]

    string_file = "Dataset," + ','.join(["KDEBN_Validation_"+ str(f) + "_" + str(p) for f in folds for p in patience])
    print(string_file)

    for file in files:
        x = experiments_helper.validate_dataset(file, [2, 3, 5, 10])
        if x is None:
            continue
        else:
            dataset, result_folder = x

        validation_scores = np.full((10, len(folds), len(patience)), np.nan)

        for (idx_fold, (train_indices, test_indices)) in enumerate(KFold(10, shuffle=True, random_state=0).split(dataset)):
            train_dataset = dataset.iloc[train_indices,:]
            test_dataset = dataset.iloc[test_indices,:]

            validation_scores[idx_fold] = test_validation_kde_network(train_dataset, test_dataset, folds, patience,
                                                                    result_folder, idx_fold)

        sum_validation = validation_scores.sum(axis=0)

        basefolder = os.path.basename(os.path.dirname(file))
        new_line = basefolder

        for idx_f, f in enumerate(folds):
            for idx_p, p in enumerate(patience):
                new_line += "," + str(sum_validation[idx_f, idx_p])

        print(new_line)
        string_file += '\n' + new_line

    with open('cv_results_kdenetwork.csv', 'w') as f:
        f.write(string_file)


if __name__ == '__main__':
    # train_crossvalidation()
    test_crossvalidation()