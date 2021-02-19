import os
import experiments_helper
import pathlib
import multiprocessing as mp
from pybnesian.graph import load
from pybnesian.models import KDENetwork


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

def run_pc_lc_kdebn(train_data, folds, patience, result_folder, idx_fold):
    fold_folder = result_folder + '/PC/KDEBN/LinearCorrelation/' + str(idx_fold)
    pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

    pdag = load(result_folder + '/PC/graph-lc-'+ str(idx_fold) + ".pickle")

    try:
        dag = pdag.to_dag()
    except ValueError:
        dag = remove_bidirected(pdag)

    kdebn = KDENetwork(dag)
    kdebn.save(fold_folder + "/000000")

def run_pc_kmi_kdebn(train_data, folds, patience, result_folder, idx_fold):
    fold_folder = result_folder + '/PC/KDEBN/KMutualInformation/' + str(idx_fold)
    pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

    pdag = load(result_folder + '/PC/graph-kmi-'+ str(idx_fold) + ".pickle")

    try:
        dag = pdag.to_dag()
    except ValueError:
        dag = remove_bidirected(pdag)

    kdebn = KDENetwork(dag)
    kdebn.save(fold_folder + "/000000")


def run_pc_kdebn(train_data, folds, patience, result_folder, idx_fold):
    run_pc_lc_kdebn(train_data, folds, patience, result_folder, idx_fold)
    run_pc_kmi_kdebn(train_data, folds, patience, result_folder, idx_fold)

def train_crossvalidation_file(file, folds):
    x = experiments_helper.validate_dataset(file, folds)
    if x is None:
        return
    else:
        dataset, result_folder = x

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    with mp.Pool(processes=experiments_helper.EVALUATION_FOLDS) as p:
        p.starmap(run_pc_kdebn, [(dataset.iloc[train_indices,:], folds, result_folder, idx_fold)
                                             for (idx_fold, (train_indices, test_indices)) in
                                             enumerate(KFold(experiments_helper.EVALUATION_FOLDS, shuffle=True, 
                                                             random_state=experiments_helper.SEED).split(dataset))]
                  )

def train_crossvalidation():
    files = experiments_helper.find_crossvalidation_datasets()

    for file in files:
        train_crossvalidation_file(file, experiments_helper.TRAINING_FOLDS)

train_crossvalidation()