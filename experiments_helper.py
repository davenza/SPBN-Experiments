import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from pybnesian.learning.scores import ValidatedLikelihood

SEED = 0
EVALUATION_FOLDS = 10
TRAINING_FOLDS = [10]
PATIENCE = [0, 5]

def find_crossvalidation_datasets():
    csv_files = glob.glob('data/**/*.csv')

    csv_files = list(filter(lambda n: not os.path.splitext(os.path.basename(n))[0].endswith("_full"), csv_files))
    csv_files = list(filter(lambda n: not os.path.splitext(os.path.basename(n))[0].endswith("_tra"), csv_files))
    csv_files = list(filter(lambda n: not os.path.splitext(os.path.basename(n))[0].endswith("_tes"), csv_files))

    csv_files = sorted(csv_files)
    return csv_files

def remove_crossvalidated_nan(dataset, folds):
    to_delete = set()

    # Outer for: Performance CV
    for (idx_fold, (train_indices, test_indices)) in enumerate(KFold(EVALUATION_FOLDS, shuffle=True, random_state=SEED).split(dataset)):
        train_data = dataset.iloc[train_indices,:]
        # Inner for: Validation CV
        for k in folds:
            vl = ValidatedLikelihood(train_data, k=k, seed=SEED)
            for (train_fold, _) in vl.cv_lik.cv:
                train_fold_pandas = train_fold.to_pandas()
                d = train_fold_pandas.columns[np.isclose(train_fold_pandas.var(), 0)].tolist()
                to_delete.update(d)

    return to_delete


def linear_dependent_features(dataset):
    to_remove = set()
    rank = np.linalg.matrix_rank(dataset)
    min_dim = min(dataset.shape[0], dataset.shape[1])

    tmp_dataset = dataset.copy()
    for i in range(min_dim - rank):
        for column in tmp_dataset.columns:
            cpy = tmp_dataset.drop(column, axis=1)
            if np.linalg.matrix_rank(cpy) == rank:
                to_remove.add(column)
                tmp_dataset = cpy
                break

    return to_remove


def validate_dataset(file, folds):
    basefolder = os.path.basename(os.path.dirname(file))

    # TODO Review bug in QSAR biodegradation Validation10_5 fold 0.
    if basefolder == "Letter" or\
            basefolder == "MFeatures" or\
            basefolder == "Musk" or\
            basefolder == "QSAR biodegradation":
        return

    result_folder = 'models/' + basefolder

    dataset = pd.read_csv(file)
    if "class" in dataset.columns:
        dataset = dataset.drop("class", axis=1)
    dataset = dataset.astype('float64')

    # Remove constant-value features.
    to_remove_features = remove_crossvalidated_nan(dataset, folds)
    dataset = dataset.drop(to_remove_features, axis=1)
    # Remove linear dependent features
    dependent_features = linear_dependent_features(dataset)
    dataset = dataset.drop(dependent_features, axis=1)

    return dataset, result_folder

def remove_bidirected(pdag):
    arcs = pdag.arcs()
    bidirected_arcs = []
    
    for arc in arcs:
        if arc[::-1] in arcs:
            bidirected_arcs.append(arc)

            arcs.remove(arc)
            arcs.remove(arc[::-1])

    for to_remove in bidirected_arcs:
        pdag.remove_arc(to_remove[0], to_remove[1])

    return pdag.to_dag()