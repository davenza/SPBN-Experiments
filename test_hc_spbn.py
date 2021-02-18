import glob
import os
import numpy as np
import experiments_helper
from pybnesian.models import load_model

from sklearn.model_selection import KFold

def test_validation_spbn(train_data, test_data, folds, patience, result_folder, idx_fold):
    test_scores = np.full((len(folds), len(patience)), np.nan)

    for idx_k, k in enumerate(folds):
        for idx_p, p in enumerate(patience):
            models_folder = result_folder + '/HillClimbing/SPBN/Validation_' + str(k) + "_" + str(p) + '/' + str(idx_fold)
            all_models = sorted(glob.glob(models_folder + '/*.pickle'))
            final_model = load_model(all_models[-1])
            final_model.fit(train_data)
            test_scores[idx_k, idx_p] = final_model.slogl(test_data)

    return test_scores

def test_crossvalidation():
    files = experiments_helper.find_crossvalidation_datasets()

    folds = experiments_helper.TRAINING_FOLDS
    patience = experiments_helper.PATIENCE

    string_file = "Dataset," + ','.join(["SPBN_Validation_" + str(f) + "_" + str(p) for f in folds for p in patience])

    print(string_file)
    for file in files:
        x = experiments_helper.validate_dataset(file, experiments_helper.TRAINING_FOLDS)
        if x is None:
            continue
        else:
            dataset, result_folder = x

        spbn_score = np.full((experiments_helper.EVALUATION_FOLDS, len(folds), len(patience)), np.nan)

        for (idx_fold, (train_indices, test_indices)) in enumerate(KFold(experiments_helper.EVALUATION_FOLDS, shuffle=True, 
                                                                   random_state=experiments_helper.SEED).split(dataset)):
            train_dataset = dataset.iloc[train_indices,:]
            test_dataset = dataset.iloc[test_indices,:]

            spbn_score[idx_fold] = test_validation_spbn(train_dataset, test_dataset, folds, patience,
                                                             result_folder, idx_fold)

        sum_spbn_score = spbn_score.sum(axis=0)

        basefolder = os.path.basename(os.path.dirname(file))
        new_line = basefolder

        for idx_f, f in enumerate(folds):
            for idx_p, p in enumerate(patience):
                new_line += "," + str(sum_spbn_score[idx_f, idx_p])

        print(new_line)

        string_file += '\n' + new_line

    with open('results_hc_spbn.csv', 'w') as f:
        f.write(string_file)

test_crossvalidation()