import glob
import os
import numpy as np
np.random.seed(0)
import pandas as pd
from pgmpy.estimators import HybridCachedHillClimbing, CachedHillClimbing
from pgmpy.estimators import CVPredictiveLikelihood, ValidationLikelihood, GaussianValidationLikelihood, \
    GaussianBicScore, BGeScore
from pgmpy.estimators.callbacks import DrawModel, SaveModel
from pgmpy.models import LinearGaussianBayesianNetwork, HybridContinuousModel

from sklearn.model_selection import KFold
import pathlib

import multiprocessing as mp
import experiments_helper


def run_validation_ckde(train_data, folds, patience, result_folder, idx_fold):
    for k in folds:
        for p in patience:
            fold_folder = result_folder + '/CKDE/Validation_' + str(k) + '_' + str(p) + '/' + str(idx_fold)
            pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

            if os.path.exists(fold_folder + '/end.lock'):
                continue

            vl = ValidationLikelihood(train_data, k=k, seed=0)
            hc = HybridCachedHillClimbing(train_data, scoring_method=vl)
            cb_draw = DrawModel(fold_folder)
            cb_save = SaveModel(fold_folder)
            bn = hc.estimate(callbacks=[cb_draw, cb_save], patience=p)
            with open(fold_folder + '/end.lock', 'w') as f:
                pass


def run_cv_ckde(train_data, folds, result_folder, idx_fold):
    for k in folds:
        fold_folder = result_folder + '/CKDE/CV_' + str(k) + '/' + str(idx_fold)
        pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

        if os.path.exists(fold_folder + '/end.lock'):
            continue

        cv = CVPredictiveLikelihood(train_data, k=k, seed=0)
        hc = HybridCachedHillClimbing(train_data, scoring_method=cv)
        cb_draw = DrawModel(fold_folder)
        cb_save = SaveModel(fold_folder)
        bn = hc.estimate(callbacks=[cb_draw, cb_save])
        with open(fold_folder + '/end.lock', 'w') as f:
            pass


def run_validation_gaussian(train_data, folds, patience, result_folder, idx_fold):
    for k in folds:
        for p in patience:
            fold_folder = result_folder + '/Gaussian/Validation_' + str(k) + '_' + str(p) + '/' + str(idx_fold)
            pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

            if os.path.exists(fold_folder + '/end.lock'):
                continue

            gv = GaussianValidationLikelihood(train_data, k=k, seed=0)
            cb_draw = DrawModel(fold_folder)
            cb_save = SaveModel(fold_folder)
            ghc = CachedHillClimbing(train_data, scoring_method=gv)
            gbn = ghc.estimate(callbacks=[cb_draw, cb_save], patience=p)
            with open(fold_folder + '/end.lock', 'w') as f:
                pass


def run_bic_gaussian(train_data, result_folder, idx_fold):
    fold_folder = result_folder + '/Gaussian/BIC/' + str(idx_fold)
    pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

    if os.path.exists(fold_folder + '/end.lock'):
        return

    bic = GaussianBicScore(train_data)
    ghc = CachedHillClimbing(train_data, scoring_method=bic)
    cb_draw = DrawModel(fold_folder)
    cb_save = SaveModel(fold_folder)
    gbn = ghc.estimate(callbacks=[cb_draw, cb_save])
    with open(fold_folder + '/end.lock', 'w') as f:
        pass


def run_bge_gaussian(train_data, result_folder, idx_fold):
    fold_folder = result_folder + '/Gaussian/BGe/' + str(idx_fold)
    pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

    if os.path.exists(fold_folder + '/end.lock'):
        return

    bge = BGeScore(train_data)
    ghc = CachedHillClimbing(train_data, scoring_method=bge)
    cb_draw = DrawModel(fold_folder)
    cb_save = SaveModel(fold_folder)
    gbn = ghc.estimate(callbacks=[cb_draw, cb_save])
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
    if not os.path.exists(result_folder + '/CKDE'):
        os.mkdir(result_folder + '/CKDE')
    if not os.path.exists(result_folder + '/Gaussian'):
        os.mkdir(result_folder + '/Gaussian')

    with mp.Pool(processes=10) as p:
        p.starmap(run_validation_ckde, [(dataset.iloc[train_indices,:], folds, patience, result_folder, idx_fold)
                                             for (idx_fold, (train_indices, test_indices)) in
                                             enumerate(KFold(10, shuffle=True, random_state=0).split(dataset))]
                  )

    with mp.Pool(processes=10) as p:
        p.starmap(run_validation_gaussian, [(dataset.iloc[train_indices,:], folds, patience, result_folder, idx_fold)
                                             for (idx_fold, (train_indices, test_indices)) in
                                             enumerate(KFold(10, shuffle=True, random_state=0).split(dataset))]
                  )

    with mp.Pool(processes=10) as p:
        p.starmap(run_bic_gaussian, [(dataset.iloc[train_indices,:], result_folder, idx_fold)
                                             for (idx_fold, (train_indices, test_indices)) in
                                             enumerate(KFold(10, shuffle=True, random_state=0).split(dataset))]
                  )
        
    with mp.Pool(processes=10) as p:
        p.starmap(run_bge_gaussian, [(dataset.iloc[train_indices,:], result_folder, idx_fold)
                                             for (idx_fold, (train_indices, test_indices)) in
                                             enumerate(KFold(10, shuffle=True, random_state=0).split(dataset))]
                  )


def train_crossvalidation():
    files = experiments_helper.find_crossvalidation_datasets()

    folds = [10]
    patience = [0, 5]

    for file in files:
        train_crossvalidation_file(file, folds, patience)


def test_validation_ckde(train_data, test_data, folds, patience, result_folder, idx_fold):
    test_scores = np.full((len(folds), len(patience)), np.nan)

    for idx_k, k in enumerate(folds):
        for idx_p, p in enumerate(patience):
            models_folder = result_folder + '/CKDE/Validation_' + str(k) + "_" + str(p) + '/' + str(idx_fold)
            all_models = sorted(glob.glob(models_folder + '/*.pkl'))
            final_model = HybridContinuousModel.load_model(all_models[-1])
            final_model.fit(train_data)
            test_scores[idx_k, idx_p] = final_model.logpdf_dataset(test_data).sum()

    return test_scores


def test_cv_ckde(train_data, test_data, folds, result_folder, idx_fold):
    test_scores = np.full((len(folds),), np.nan)

    for idx_k, k in enumerate(folds):
        models_folder = result_folder + '/CKDE/CV_' + str(k) + '/' + str(idx_fold)
        all_models = sorted(glob.glob(models_folder + '/*.pkl'))
        final_model = HybridContinuousModel.load_model(all_models[-1])
        final_model.fit(train_data)
        test_scores[idx_k] = final_model.logpdf_dataset(test_data).sum()

    return test_scores


def test_validation_gaussian(train_data, test_data, folds, patience, result_folder, idx_fold):
    test_scores = np.full((len(folds), len(patience)), np.nan)

    for idx_k, k in enumerate(folds):
        for idx_p, p in enumerate(patience):
            models_folder = result_folder + '/Gaussian/Validation_' + str(k) + "_" + str(p) + '/' + str(idx_fold)
            all_models = sorted(glob.glob(models_folder + '/*.pkl'))
            final_model = LinearGaussianBayesianNetwork.load_model(all_models[-1])
            final_model.fit(train_data)
            test_scores[idx_k, idx_p] = final_model.logpdf_dataset(test_data).sum()

    return test_scores


def test_bic_gaussian(train_data, test_data, result_folder, idx_fold):
    models_folder = result_folder + '/Gaussian/BIC/' + str(idx_fold)
    all_models = sorted(glob.glob(models_folder + '/*.pkl'))
    final_model = LinearGaussianBayesianNetwork.load_model(all_models[-1])
    final_model.fit(train_data)

    return final_model.logpdf_dataset(test_data).sum()


def test_bge_gaussian(train_data, test_data, result_folder, idx_fold):
    models_folder = result_folder + '/Gaussian/BGe/' + str(idx_fold)
    all_models = sorted(glob.glob(models_folder + '/*.pkl'))
    final_model = LinearGaussianBayesianNetwork.load_model(all_models[-1])
    final_model.fit(train_data)

    return final_model.logpdf_dataset(test_data).sum()


def test_crossvalidation():
    files = experiments_helper.find_crossvalidation_datasets()

    folds = [2, 3, 5, 10]
    patience = [0, 5]

    string_file = "Dataset," + ','.join(["CKDE_Validation_"+ str(f) + "_" + str(p) for f in folds for p in patience])\
                   + ',' + ','.join(["Gaussian_Validation_"+ str(f) + "_" + str(p) for f in folds for p in patience]) +\
                  ",BIC,BGe"

    print(string_file)
    for file in files:
        x = experiments_helper.validate_dataset(file, [2, 3, 5, 10])
        if x is None:
            continue
        else:
            dataset, result_folder = x

        validation_ckde = np.full((10, len(folds), len(patience)), np.nan)
        cv_ckde = np.full((10, len(folds)), np.nan)
        validation_gaussian = np.full((10, len(folds), len(patience)), np.nan)
        bic_gaussian = np.full((10,), np.nan)
        bge_gaussian = np.full((10,), np.nan)

        for (idx_fold, (train_indices, test_indices)) in enumerate(KFold(10, shuffle=True, random_state=0).split(dataset)):
            train_dataset = dataset.iloc[train_indices,:]
            test_dataset = dataset.iloc[test_indices,:]

            validation_ckde[idx_fold] = test_validation_ckde(train_dataset, test_dataset, folds, patience,
                                                             result_folder, idx_fold)
            # cv_ckde[idx_fold] = test_cv_ckde(train_dataset, test_dataset, folds, result_folder, idx_fold)
            validation_gaussian[idx_fold] = test_validation_gaussian(train_dataset, test_dataset, folds, patience,
                                                                     result_folder, idx_fold)
            bic_gaussian[idx_fold] = test_bic_gaussian(train_dataset, test_dataset, result_folder, idx_fold)
            bge_gaussian[idx_fold] = test_bge_gaussian(train_dataset, test_dataset, result_folder, idx_fold)

        sum_validation_ckde = validation_ckde.sum(axis=0)
        sum_validation_gaussian = validation_gaussian.sum(axis=0)
        sum_bic = bic_gaussian.sum(axis=0)
        sum_bge = bge_gaussian.sum(axis=0)

        basefolder = os.path.basename(os.path.dirname(file))
        new_line = basefolder

        for idx_f, f in enumerate(folds):
            for idx_p, p in enumerate(patience):
                new_line += "," + str(sum_validation_ckde[idx_f, idx_p])
        for idx_f, f in enumerate(folds):
            for idx_p, p in enumerate(patience):
                new_line += "," + str(sum_validation_gaussian[idx_f, idx_p])

        new_line += "," + str(sum_bic)
        new_line += "," + str(sum_bge)
        print(new_line)

        string_file += '\n' + new_line

    with open('cv_results_bn.csv', 'w') as f:
        f.write(string_file)

def run_holdout():
    pass



if __name__ == '__main__':
    # train_crossvalidation()
    # train_crossvalidation_file('data/MFeatures/mfeat.csv', [2, 3, 5, 10], [0, 5])
    test_crossvalidation()

    # files = find_crossvalidation_datasets()
    # print(str(len(files)) + " CV datasets found")
    # for file in files:
    #     basefolder = os.path.basename(os.path.dirname(file))
    #     result_folder = 'models/' + basefolder
    #     print("Dataset " + basefolder)
    #     print("----------------------------------")
    #
    #     dataset = pd.read_csv(file)
    #     dataset = dataset.drop("class", axis=1, errors="ignore")
    #     dataset = dataset.astype('float64')
    #
    #     to_remove = set()
    #     to_remove.update(remove_crossvalidated_nan(dataset, [2, 3, 5, 10]))
    #
    #     print("Removing columns nan: " + str(to_remove))
    #     dataset = dataset.drop(to_remove, axis=1)
    #
    #     print("Dataset " + str(basefolder) + " " + str(len(dataset)) + " x " + str(dataset.shape[1]))
    #
    #     rank = np.linalg.matrix_rank(dataset)
    #     print("Rank matrix " + str(rank))
    #
    #     min_dim = min(dataset.shape[0], dataset.shape[1])
    #
    #     if rank < min_dim:
    #         rank_columns = linear_dependent_features(dataset)
    #         print("Rank removing columns: " + str(rank_columns))
    #         dataset = dataset.drop(rank_columns, axis=1)
    #         print("Final dataset " + str(basefolder) + " " + str(len(dataset)) + " x " + str(dataset.shape[1]))
    #         print("Final rank matrix " + str(np.linalg.matrix_rank(dataset)))
    #
    #     print()