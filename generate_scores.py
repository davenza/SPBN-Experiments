import os
import glob
import numpy as np
import pandas as pd
import run_experiments

from pgmpy.models import HybridContinuousModel
from pgmpy.estimators import ValidationLikelihood

from sklearn.model_selection import KFold

import multiprocessing as mp

def load_dataset(basename):
    data_file = glob.glob('data/' + basename + '/*.csv')
    data_file = list(filter(lambda n: not os.path.splitext(os.path.basename(n))[0].endswith("_full"), data_file))
    data_file = list(filter(lambda n: not os.path.splitext(os.path.basename(n))[0].endswith("_tra"), data_file))
    data_file = list(filter(lambda n: not os.path.splitext(os.path.basename(n))[0].endswith("_tes"), data_file))
    data_file = data_file[0]

    dataset = pd.read_csv(data_file)
    if "class" in dataset.columns:
        dataset = dataset.drop("class", axis=1)
    dataset = dataset.astype('float64')

    to_remove_features = run_experiments.remove_crossvalidated_nan(dataset, [2, 3, 5, 10])
    dataset = dataset.drop(to_remove_features, axis=1)

    dependent_features = run_experiments.linear_dependent_features(dataset)
    dataset = dataset.drop(dependent_features, axis=1)

    return dataset

def find_op(previous_hcm, new_hcm):
    previous_edges = set(previous_hcm.edges)
    new_edges = set(new_hcm.edges)
    if set(previous_edges) != set(new_edges):
        added_set = new_edges - previous_edges
        removed_set = previous_edges - new_edges
        if added_set:
            a_source, a_dest = added_set.pop()
            if removed_set:
                r_source, r_dest = removed_set.pop()
                if a_source != r_dest and a_dest != r_source:
                    raise ValueError("Error in flip operator")

                return ("flip", r_source, r_dest)
            else:
                return ("+", a_source, a_dest)
        else:
            r_source, r_dest = removed_set.pop()
            return ("-", r_source, r_dest)
    else:
        for (n, new_type) in new_hcm.node_type.items():
            if previous_hcm.node_type[n] != new_type:
                return ("type", n, new_type)


def save_model_scores(model_folder, dataset, typ, k, patience):
    if typ != "Validation":
        return

    models = sorted(glob.glob(model_folder + '/*.pkl'))
    fold_index = int(os.path.basename(model_folder))

    cv_indices = list(KFold(10, shuffle=True, random_state=0).split(dataset))

    train_data = dataset.iloc[cv_indices[fold_index][0], :]

    vl = ValidationLikelihood(train_data, k=k, seed=0)

    file_str = "Iteration,TrainScore,ValidationScore,Op,OpSource,OpDest,DeltaTrain,DeltaValidation"
    hcm = None
    train_score = np.nan
    validation_score = np.nan

    for model in models:
        previous_hcm = hcm
        previous_train_score = train_score
        previous_validation_score = validation_score

        print("Executing " + model)
        filename = os.path.splitext(os.path.basename(model))[0]
        iter_no = int(filename)

        hcm = HybridContinuousModel.load_model(model)

        if previous_hcm is not None and model != models[-1]:
            op = find_op(previous_hcm, hcm)
        else:
            op = None

        train_score = 0
        validation_score = 0

        for n in hcm.nodes:
            train_score += vl.local_score(n, hcm.get_parents(n), hcm.node_type[n], hcm.node_type)
            validation_score += vl.validation_local_score(n, hcm.get_parents(n), hcm.node_type[n], hcm.node_type)

        file_str += '\n' + str(iter_no) + "," + str(train_score) + "," + str(validation_score)
        if op is None:
            file_str += ',NA,NA,NA,NA,NA'
        else:
            delta_train_score = train_score - previous_train_score
            delta_validation_score = validation_score - previous_validation_score
            file_str += ',' + op[0] + ',' + op[1] + ',' + str(op[2]) + ',' + str(delta_train_score) + \
                        ',' + str(delta_validation_score)

    with open(model_folder + '/scores.csv', 'w') as f:
        f.write(file_str)

print("Number of models: " + str(len(glob.glob('models/**/*.pkl', recursive=True))))

datasets = glob.glob('models/*')

for data in datasets:
    print("Data " + os.path.basename(data))

    dataset = load_dataset(os.path.basename(data))

    cv_folder = data + '/CKDE/*'
    cv_types = glob.glob(cv_folder)

    for cv_type in cv_types:
        typ, k, patience = os.path.basename(cv_type).split('_')

        index_cv_folder = cv_type + '/*'
        model_folders = glob.glob(index_cv_folder)

        with mp.Pool(processes=len(model_folders)) as p:
            p.starmap(save_model_scores, [(model_folder, dataset, typ, int(k), int(patience))
                                            for model_folder in model_folders]
                      )

        # for model_folder in model_folders:
        #     save_model_scores(model_folder, dataset, typ, int(k), int(patience))
