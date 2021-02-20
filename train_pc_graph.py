import os
import experiments_helper
import pathlib
import multiprocessing as mp
from pybnesian.learning.algorithms import PC
from pybnesian.learning.independences import LinearCorrelation, KMutualInformation

from sklearn.model_selection import KFold

def run_pc_graph(train_data, result_folder, idx_fold):
    fold_folder = result_folder + '/PC'

    pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)
    
    pc = PC()
    
    if not os.path.exists(fold_folder + '/graph-lc-' + str(idx_fold) + '.pickle'):
        lc = LinearCorrelation(train_data)
        
        graph_lc = pc.estimate(lc)
        graph_lc.save(fold_folder + '/graph-lc-' + str(idx_fold))

    if not os.path.exists(fold_folder + '/graph-kmi-' + str(idx_fold) + '.pickle'):
        kmi = KMutualInformation(train_data, k=25, seed=experiments_helper.SEED)
        
        graph_kmi = pc.estimate(kmi)
        graph_kmi.save(fold_folder + '/graph-kmi-' + str(idx_fold))

def train_crossvalidation_file(file, folds):
    x = experiments_helper.validate_dataset(file, folds)
    if x is None:
        return
    else:
        dataset, result_folder = x

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    with mp.Pool(processes=experiments_helper.EVALUATION_FOLDS) as p:
        p.starmap(run_pc_graph, [(dataset.iloc[train_indices,:], result_folder, idx_fold)
                                             for (idx_fold, (train_indices, test_indices)) in
                                             enumerate(KFold(experiments_helper.EVALUATION_FOLDS, shuffle=True, 
                                                             random_state=experiments_helper.SEED).split(dataset))]
                  )

def train_crossvalidation():
    files = experiments_helper.find_crossvalidation_datasets()

    for file in files:
        train_crossvalidation_file(file, experiments_helper.TRAINING_FOLDS)

train_crossvalidation()