import os
import glob
import experiments_helper
import pathlib
import multiprocessing as mp
from pybnesian.learning.algorithms import GreedyHillClimbing
from pybnesian.learning.algorithms.callbacks import SaveModel
from pybnesian.learning.operators import ArcOperatorSet
from pybnesian.learning.scores import ValidatedLikelihood, BIC, BGe
from pybnesian.models import GaussianNetwork

from sklearn.model_selection import KFold

def run_validation_gaussian(train_data, folds, patience, result_folder, idx_fold):
    hc = GreedyHillClimbing()
    arc_set = ArcOperatorSet()

    for k in folds:
        vl = ValidatedLikelihood(train_data, k=k, seed=0)

        for p in patience:
            fold_folder = result_folder + '/HillClimbing/Gaussian/Validation_' + str(k) + '_' + str(p) + '/' + str(idx_fold)
            pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

            if os.path.exists(fold_folder + '/end.lock'):
                continue

            cb_save = SaveModel(fold_folder)
            start_model = GaussianNetwork(list(train_data.columns.values))
            bn = hc.estimate(arc_set, vl, start_model, callback=cb_save, patience=p, verbose=True)
            iters = sorted(glob.glob(fold_folder + '/*.pickle'))
            last_file = os.path.basename(iters[-1])
            number = int(os.path.splitext(last_file)[0])
            bn.save(fold_folder + '/' + str(number+1).zfill(6) + ".pickle")
            with open(fold_folder + '/end.lock', 'w') as f:
                pass

def run_bic_gaussian(train_data, result_folder, idx_fold):
    fold_folder = result_folder + '/HillClimbing/Gaussian/BIC/' + str(idx_fold)
    pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

    if os.path.exists(fold_folder + '/end.lock'):
        return

    hc = GreedyHillClimbing()
    arc_set = ArcOperatorSet()

    bic = BIC(train_data)

    cb_save = SaveModel(fold_folder)
    start_model = GaussianNetwork(list(train_data.columns.values))
    
    bn = hc.estimate(arc_set, bic, start_model, callback=cb_save, verbose=True)
    iters = sorted(glob.glob(fold_folder + '/*.pickle'))
    last_file = os.path.basename(iters[-1])
    number = int(os.path.splitext(last_file)[0])
    bn.save(fold_folder + '/' + str(number+1).zfill(6) + ".pickle")
    with open(fold_folder + '/end.lock', 'w') as f:
        pass

def run_bge_gaussian(train_data, result_folder, idx_fold):
    fold_folder = result_folder + '/HillClimbing/Gaussian/BGe/' + str(idx_fold)
    pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

    if os.path.exists(fold_folder + '/end.lock'):
        return

    hc = GreedyHillClimbing()
    arc_set = ArcOperatorSet()

    bge = BGe(train_data)

    cb_save = SaveModel(fold_folder)
    start_model = GaussianNetwork(list(train_data.columns.values))
    
    bn = hc.estimate(arc_set, bge, start_model, callback=cb_save, verbose=True)
    iters = sorted(glob.glob(fold_folder + '/*.pickle'))
    last_file = os.path.basename(iters[-1])
    number = int(os.path.splitext(last_file)[0])
    bn.save(fold_folder + '/' + str(number+1).zfill(6) + ".pickle")
    with open(fold_folder + '/end.lock', 'w') as f:
        pass

def train_crossvalidation_file(file, folds, patience):
    x = experiments_helper.validate_dataset(file, folds)
    if x is None:
        return
    else:
        dataset, result_folder = x

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    print(file)

    with mp.Pool(processes=experiments_helper.EVALUATION_FOLDS) as p:
        p.starmap(run_validation_gaussian, [(dataset.iloc[train_indices,:], folds, patience, result_folder, idx_fold)
                                             for (idx_fold, (train_indices, test_indices)) in
                                             enumerate(KFold(experiments_helper.EVALUATION_FOLDS, shuffle=True, 
                                                             random_state=experiments_helper.SEED).split(dataset))]
                  )

    with mp.Pool(processes=experiments_helper.EVALUATION_FOLDS) as p:
        p.starmap(run_bic_gaussian, [(dataset.iloc[train_indices,:], result_folder, idx_fold)
                                             for (idx_fold, (train_indices, test_indices)) in
                                             enumerate(KFold(experiments_helper.EVALUATION_FOLDS, shuffle=True, 
                                                             random_state=0).split(dataset))]
                  )
    
    with mp.Pool(processes=experiments_helper.EVALUATION_FOLDS) as p:
        p.starmap(run_bge_gaussian, [(dataset.iloc[train_indices,:], result_folder, idx_fold)
                                             for (idx_fold, (train_indices, test_indices)) in
                                             enumerate(KFold(experiments_helper.EVALUATION_FOLDS,
                                                             shuffle=True, random_state=0).split(dataset))]
                  )

def train_crossvalidation():
    files = experiments_helper.find_crossvalidation_datasets()

    for file in files:
        train_crossvalidation_file(file, experiments_helper.TRAINING_FOLDS, experiments_helper.PATIENCE)

train_crossvalidation()