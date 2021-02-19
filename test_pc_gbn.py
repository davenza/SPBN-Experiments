import glob
import os
import numpy as np
import experiments_helper
from pybnesian import load

from sklearn.model_selection import KFold

def test_pc_lc_gbn(train_data, test_data, result_folder, idx_fold):
    models_folder = result_folder + '/PC/Gaussian/LinearCorrelation/' + str(idx_fold)
    all_models = sorted(glob.glob(models_folder + '/*.pickle'))
    final_model = load(all_models[-1])
    final_model.fit(train_data)
    return final_model.slogl(test_data)

def test_pc_kmi_gbn(train_data, test_data, result_folder, idx_fold):
    models_folder = result_folder + '/PC/Gaussian/KMutualInformation/' + str(idx_fold)
    all_models = sorted(glob.glob(models_folder + '/*.pickle'))
    final_model = load(all_models[-1])
    final_model.fit(train_data)
    return final_model.slogl(test_data)

def test_crossvalidation():
    files = experiments_helper.find_crossvalidation_datasets()

    string_file = "Dataset,GBN_PC_LC,GBN_PC_KMI"

    print(string_file)
    for file in files:
        x = experiments_helper.validate_dataset(file, experiments_helper.TRAINING_FOLDS)
        if x is None:
            continue
        else:
            dataset, result_folder = x

        gbn_lc_score = np.full((experiments_helper.EVALUATION_FOLDS,), np.nan)
        gbn_kmi_score = np.full((experiments_helper.EVALUATION_FOLDS,), np.nan)

        for (idx_fold, (train_indices, test_indices)) in enumerate(KFold(experiments_helper.EVALUATION_FOLDS, shuffle=True, 
                                                                   random_state=experiments_helper.SEED).split(dataset)):
            train_dataset = dataset.iloc[train_indices,:]
            test_dataset = dataset.iloc[test_indices,:]

            gbn_lc_score[idx_fold] = test_pc_lc_gbn(train_dataset, test_dataset, result_folder, idx_fold)
            gbn_kmi_score[idx_fold] = test_pc_kmi_gbn(train_dataset, test_dataset, result_folder, idx_fold)

        sum_gbn_lc_score = gbn_lc_score.sum(axis=0)
        sum_gbn_kmi_score = gbn_kmi_score.sum(axis=0)

        basefolder = os.path.basename(os.path.dirname(file))
        new_line = basefolder + "," + str(sum_gbn_lc_score) + "," + str(sum_gbn_pc_kmi)

        print(new_line)

        string_file += '\n' + new_line

    with open('results_pc_gbn.csv', 'w') as f:
        f.write(string_file)

test_crossvalidation()