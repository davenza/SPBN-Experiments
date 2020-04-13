import os
import experiments_helper
import multiprocessing as mp
from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture
import pathlib
from joblib import dump, load
import numpy as np

def run_gmm_components(train_data, components, result_folder, idx_fold):
    for k in components:
        fold_folder = result_folder + '/GMM/Components_' + str(k) + '/' + str(idx_fold)
        pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

        if os.path.exists(fold_folder + '/end.lock'):
            continue

        gmm = GaussianMixture(k, n_init=3, random_state=0)
        gmm.fit(train_data)
        dump(gmm, fold_folder + '/model.pkl')
        with open(fold_folder + '/end.lock', 'w') as f:
            pass


def train_crossvalidation_file(file, components):
    print("Training " + str(file))

    x = experiments_helper.validate_dataset(file, [2, 3, 5, 10])
    if x is None:
        return
    else:
        dataset, result_folder = x

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    if not os.path.exists(result_folder + '/GMM'):
        os.mkdir(result_folder + '/GMM')

    # for (idx_fold, (train_indices, test_indices)) in enumerate(KFold(10, shuffle=True, random_state=0).split(dataset)):
    #     run_gmm_components(dataset.iloc[train_indices,:], components, result_folder, idx_fold)

    with mp.Pool(processes=10) as p:
        p.starmap(run_gmm_components, [(dataset.iloc[train_indices,:], components, result_folder, idx_fold)
                                             for (idx_fold, (train_indices, test_indices)) in
                                             enumerate(KFold(10, shuffle=True, random_state=0).split(dataset))]
                  )

def train_crossvalidation():
    files = experiments_helper.find_crossvalidation_datasets()

    components = list(range(2, 11)) + list(range(12, 31, 2))

    for file in files:
        train_crossvalidation_file(file, components)


def test_gmm(test_data, components, result_folder, idx_fold):
    test_score = np.full((len(components),), np.nan)

    for idx_k, k in enumerate(components):
        model_name = result_folder + '/GMM/Components_' + str(k) + '/' + str(idx_fold) + '/model.pkl'

        gmm = load(model_name)
        test_score[idx_k] = gmm.score_samples(test_data).sum()

    return test_score

def test_crossvalidation():
    files = experiments_helper.find_crossvalidation_datasets()

    components = list(range(2, 11)) + list(range(12, 31, 2))

    string_file = "Dataset," + ','.join(["GMM_"+ str(k) for k in components])

    print(string_file)

    for file in files:
        x = experiments_helper.validate_dataset(file, [2, 3, 5, 10])
        if x is None:
            continue
        else:
            dataset, result_folder = x

        gmm = np.full((10, len(components)), np.nan)

        for (idx_fold, (train_indices, test_indices)) in enumerate(
                KFold(10, shuffle=True, random_state=0).split(dataset)):
            test_dataset = dataset.iloc[test_indices, :]

            gmm[idx_fold] = test_gmm(test_dataset, components, result_folder, idx_fold)

        sum_gmm = gmm.sum(axis=0)
        basefolder = os.path.basename(os.path.dirname(file))
        new_line = basefolder

        for idx_k, k in enumerate(components):
            new_line += "," + str(sum_gmm[idx_k])

        print(new_line)
        string_file += '\n' + new_line

    with open('cv_results_gmm.csv', 'w') as f:
        f.write(string_file)

if __name__ == '__main__':
    # train_crossvalidation()
    test_crossvalidation()