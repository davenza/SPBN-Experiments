import os
import experiments_helper
import multiprocessing as mp
import pathlib

import numpy as np
from sklearn.model_selection import KFold

from scipy.stats import gaussian_kde
import pickle

def run_kde_components(train_data, test_data, result_folder, idx_fold):
    print("idx_fold " + str(idx_fold))
    fold_folder = result_folder + '/KDE/' + str(idx_fold)
    pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

    if os.path.exists(fold_folder + '/end.lock'):
        return


    kde = gaussian_kde(train_data.to_numpy().T)
    logpdf = kde.logpdf(test_data.to_numpy().T).sum()

    with open(fold_folder + '/result.pkl', 'wb') as f:
        pickle.dump(logpdf, f)

    with open(fold_folder + '/end.lock', 'w') as f:
        pass


def train_crossvalidation_file(file):
    x = experiments_helper.validate_dataset(file, [2, 3, 5, 10])
    if x is None:
        return
    else:
        dataset, result_folder = x

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    if not os.path.exists(result_folder + '/KDE'):
        os.mkdir(result_folder + '/KDE')

    # for (idx_fold, (train_indices, test_indices)) in enumerate(KFold(10, shuffle=True, random_state=0).split(dataset)):
    #     run_kde_components(dataset.iloc[train_indices,:], dataset.iloc[test_indices,:], result_folder, idx_fold)
    #
    with mp.Pool(processes=10) as p:
        p.starmap(run_kde_components, [(dataset.iloc[train_indices,:], dataset.iloc[test_indices,:], result_folder, idx_fold)
                                             for (idx_fold, (train_indices, test_indices)) in
                                             enumerate(KFold(10, shuffle=True, random_state=0).split(dataset))]
                  )

def train_crossvalidation():
    files = experiments_helper.find_crossvalidation_datasets()

    for file in files:
        train_crossvalidation_file(file)

def test_crossvalidation():
    files = experiments_helper.find_crossvalidation_datasets()

    kde_scores = np.full((10,), np.nan)

    string_file = "Dataset,KDE"
    print(string_file)

    for file in files:
        x = experiments_helper.validate_dataset(file, [2, 3, 5, 10])
        if x is None:
            continue
        else:
            dataset, result_folder = x

        for i in range(10):
            with open(result_folder + '/KDE/' + str(i) + '/result.pkl', 'rb') as f:
                s = pickle.load(f)
                kde_scores[i] = s

        basefolder = os.path.basename(os.path.dirname(file))
        new_line = basefolder + ',' + str(kde_scores.sum())
        print(new_line)
        string_file += '\n' + new_line

    with open('cv_results_kde.csv', 'w') as f:
        f.write(string_file)


if __name__ == '__main__':
    # train_crossvalidation()
    test_crossvalidation()