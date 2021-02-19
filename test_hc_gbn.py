import glob
import os
import numpy as np
import experiments_helper
from pybnesian import load

from sklearn.model_selection import KFold

def test_validation_gaussian(train_data, test_data, folds, patience, result_folder, idx_fold):
    test_scores = np.full((len(folds), len(patience)), np.nan)

    for idx_k, k in enumerate(folds):
        for idx_p, p in enumerate(patience):
            models_folder = result_folder + '/HillClimbing/Gaussian/Validation_' + str(k) + "_" + str(p) + '/' + str(idx_fold)
            all_models = sorted(glob.glob(models_folder + '/*.pickle'))
            final_model = load(all_models[-1])
            final_model.fit(train_data)
            test_scores[idx_k, idx_p] = final_model.slogl(test_data)

    return test_scores


def test_bic_gaussian(train_data, test_data, result_folder, idx_fold):
    models_folder = result_folder + '/HillClimbing/Gaussian/BIC/' + str(idx_fold)
    all_models = sorted(glob.glob(models_folder + '/*.pickle'))
    final_model = load(all_models[-1])
    final_model.fit(train_data)

    return final_model.slogl(test_data)

def test_bge_gaussian(train_data, test_data, result_folder, idx_fold):
    models_folder = result_folder + 'HillClimbing/Gaussian/BGe/' + str(idx_fold)
    all_models = sorted(glob.glob(models_folder + '/*.pickle'))
    final_model = load(all_models[-1])
    final_model.fit(train_data)

    return final_model.logpdf_dataset(test_data).sum()

def test_crossvalidation():
    files = experiments_helper.find_crossvalidation_datasets()

    folds = experiments_helper.TRAINING_FOLDS
    patience = experiments_helper.PATIENCE

    string_file = "Dataset," + ','.join(["GBN_Validation_"+ str(f) + "_" + str(p) for f in folds for p in patience]) +\
                    ",BIC,BGe"

    print(string_file)
    for file in files:
        x = experiments_helper.validate_dataset(file, experiments_helper.TRAINING_FOLDS)
        if x is None:
            continue
        else:
            dataset, result_folder = x

        validation_gaussian = np.full((experiments_helper.EVALUATION_FOLDS, len(folds), len(patience)), np.nan)
        bic_gaussian = np.full((experiments_helper.EVALUATION_FOLDS,), np.nan)
        bge_gaussian = np.full((experiments_helper.EVALUATION_FOLDS,), np.nan)

        for (idx_fold, (train_indices, test_indices)) in enumerate(KFold(experiments_helper.EVALUATION_FOLDS, shuffle=True, 
                                                                   random_state=experiments_helper.SEED).split(dataset)):
            train_dataset = dataset.iloc[train_indices,:]
            test_dataset = dataset.iloc[test_indices,:]

            validation_gaussian[idx_fold] = test_validation_gaussian(train_dataset, test_dataset, folds, patience,
                                                                     result_folder, idx_fold)
            bic_gaussian[idx_fold] = test_bic_gaussian(train_dataset, test_dataset, result_folder, idx_fold)
            bge_gaussian[idx_fold] = test_bge_gaussian(train_dataset, test_dataset, result_folder, idx_fold)

        sum_validation_gaussian = validation_gaussian.sum(axis=0)
        sum_bic = bic_gaussian.sum(axis=0)
        sum_bge = bge_gaussian.sum(axis=0)

        basefolder = os.path.basename(os.path.dirname(file))
        new_line = basefolder

        for idx_f, f in enumerate(folds):
            for idx_p, p in enumerate(patience):
                new_line += "," + str(sum_validation_gaussian[idx_f, idx_p])
        
        new_line += "," + str(sum_bic)
        new_line += "," + str(sum_bge)
        print(new_line)

        string_file += '\n' + new_line

    with open('results_hc_gbn.csv', 'w') as f:
        f.write(string_file)

test_crossvalidation()