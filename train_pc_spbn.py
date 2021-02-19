import os
import experiments_helper
import pathlib
import multiprocessing as mp
from pybnesian import load
from pybnesian.learning.algorithms import GreedyHillClimbing, PC
from pybnesian.learning.algorithms.callbacks import SaveModel
from pybnesian.learning.operators import OperatorPool, ArcOperatorSet, ChangeNodeTypeSet
from pybnesian.learning.scores import ValidatedLikelihood
from pybnesian.learning.scores import LinearCorrelation, KMutualInformation
from pybnesian.models import SemiparametricBN


from sklearn.model_selection import KFold

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

def run_pc_lc_spbn(train_data, folds, patience, result_folder, idx_fold):
    hc = GreedyHillClimbing()
    change_node_type = ChangeNodeTypeSet()

    pdag = load(result_folder + '/PC/graph-lc-'+ str(idx_fold) + ".pickle")

    try:
        dag = pdag.to_dag()
    except ValueError:
        dag = remove_bidirected(pdag)

    for k in folds:
        vl = ValidatedLikelihood(train_data, k=k, seed=experiments_helper.SEED)

        for p in patience:
            fold_folder = result_folder + '/PC/SPBN/LinearCorrelation/Validation_' + str(k) + '_' + str(p) + '/' + str(idx_fold)
            pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

            if os.path.exists(fold_folder + '/end.lock'):
                continue

            cb_save = SaveModel(fold_folder)
            start_model = SemiparametricBN(dag)

            bn = hc.estimate(change_node_type, vl, start_model, callback=cb_save, patience=p)
            with open(fold_folder + '/end.lock', 'w') as f:
                pass

def run_pc_kmi_spbn(train_data, folds, patience, result_folder, idx_fold):
    hc = GreedyHillClimbing()
    change_node_type = ChangeNodeTypeSet()

    pdag = load(result_folder + '/PC/graph-kmi-'+ str(idx_fold) + ".pickle")

    try:
        dag = pdag.to_dag()
    except ValueError:
        dag = remove_bidirected(pdag)

    for k in folds:
        vl = ValidatedLikelihood(train_data, k=k, seed=experiments_helper.SEED)

        for p in patience:
            fold_folder = result_folder + '/PC/SPBN/KMutualInformation/Validation_' + str(k) + '_' + str(p) + '/' + str(idx_fold)
            pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

            if os.path.exists(fold_folder + '/end.lock'):
                continue

            cb_save = SaveModel(fold_folder)
            start_model = SemiparametricBN(dag)

            bn = hc.estimate(change_node_type, vl, start_model, callback=cb_save, patience=p)
            with open(fold_folder + '/end.lock', 'w') as f:
                pass

def run_pc_spbn(train_data, folds, patience, result_folder, idx_fold):
    run_pc_lc_spbn(train_data, folds, patience, result_folder, idx_fold)
    run_pc_kmi_spbn(train_data, folds, patience, result_folder, idx_fold)

def train_crossvalidation_file(file, folds, patience):
    x = experiments_helper.validate_dataset(file, folds)
    if x is None:
        return
    else:
        dataset, result_folder = x

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    with mp.Pool(processes=experiments_helper.EVALUATION_FOLDS) as p:
        p.starmap(run_pc_spbn, [(dataset.iloc[train_indices,:], folds, patience, result_folder, idx_fold)
                                             for (idx_fold, (train_indices, test_indices)) in
                                             enumerate(KFold(experiments_helper.EVALUATION_FOLDS, shuffle=True, 
                                                             random_state=experiments_helper.SEED).split(dataset))]
                  )

def train_crossvalidation():
    files = experiments_helper.find_crossvalidation_datasets()

    for file in files:
        train_crossvalidation_file(file, experiments_helper.TRAINING_FOLDS, experiments_helper.PATIENCE)

train_crossvalidation()