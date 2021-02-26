import os
import experiments_helper
import pathlib
import multiprocessing as mp
from pybnesian import load
from pybnesian.models import GaussianNetwork
from sklearn.model_selection import KFold


def run_pc_lc_gbn(result_folder, idx_fold):
    fold_folder = result_folder + '/PC/Gaussian/LinearCorrelation/' + str(idx_fold)
    pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

    pdag = load(result_folder + '/PC/graph-lc-'+ str(idx_fold) + ".pickle")

    try:
        dag = pdag.to_dag()
    except ValueError:
        dag = experiments_helper.remove_bidirected(pdag)

    gbn = GaussianNetwork(dag)
    gbn.save(fold_folder + "/000000")

def run_pc_rcot_gbn(result_folder, idx_fold):
    fold_folder = result_folder + '/PC/Gaussian/RCoT/' + str(idx_fold)
    pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

    pdag = load(result_folder + '/PC/graph-rcot-'+ str(idx_fold) + ".pickle")

    try:
        dag = pdag.to_dag()
    except ValueError:
        dag = experiments_helper.remove_bidirected(pdag)

    gbn = GaussianNetwork(dag)
    gbn.save(fold_folder + "/000000")


def run_pc_gbn(result_folder, idx_fold):
    run_pc_lc_gbn(result_folder, idx_fold)
    run_pc_rcot_gbn(result_folder, idx_fold)

def train_crossvalidation_file(file, folds):
    x = experiments_helper.validate_dataset(file, folds)
    if x is None:
        return
    else:
        dataset, result_folder = x

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    with mp.Pool(processes=experiments_helper.EVALUATION_FOLDS) as p:
        p.starmap(run_pc_gbn, [(result_folder, idx_fold)
                                for (idx_fold, (train_indices, test_indices)) in
                                enumerate(KFold(experiments_helper.EVALUATION_FOLDS, shuffle=True, 
                                                random_state=experiments_helper.SEED).split(dataset))]
                  )

def train_crossvalidation():
    files = experiments_helper.find_crossvalidation_datasets()

    for file in files:
        train_crossvalidation_file(file, experiments_helper.TRAINING_FOLDS)

train_crossvalidation()