import numpy as np
np.random.seed(0)
import pandas as pd
import pathlib
import glob
from pybnesian import load
import experiments_helper

ecoli70_true = load('ecoli70.pickle')
magic_niab_true = load('magic_niab.pickle')
magic_irri_true = load('magic_irri.pickle')
arth150_true = load('arth150.pickle')

ecoli70_200 = pd.read_csv("ecoli70_200.csv")
ecoli70_2000 = pd.read_csv("ecoli70_2000.csv")
ecoli70_10000 = pd.read_csv("ecoli70_10000.csv")
ecoli70_test = pd.read_csv("ecoli70_test.csv")

magic_niab_200 = pd.read_csv("magic_niab_200.csv")
magic_niab_2000 = pd.read_csv("magic_niab_2000.csv")
magic_niab_10000 = pd.read_csv("magic_niab_10000.csv")
magic_niab_test = pd.read_csv("magic_niab_test.csv")

magic_irri_200 = pd.read_csv("magic_irri_200.csv")
magic_irri_2000 = pd.read_csv("magic_irri_2000.csv")
magic_irri_10000 = pd.read_csv("magic_irri_10000.csv")
magic_irri_test = pd.read_csv("magic_irri_test.csv")

arth150_200 = pd.read_csv("arth150_200.csv")
arth150_2000 = pd.read_csv("arth150_2000.csv")
arth150_10000 = pd.read_csv("arth150_10000.csv")
arth150_test = pd.read_csv("arth150_test.csv")


def compare_models(true_model, trained_models_folder, training_data, test_data, patience):
    ground_truth_slogl = true_model.slogl(test_data)

    print("Ground truth loglik: " + str(ground_truth_slogl))
    print("SPBN results:")
    for p in patience:
        ckde_folder = trained_models_folder + '/HillClimbing/SPBN_CKDE/' + str(p)

        all_models = sorted(glob.glob(ckde_folder + '/*.pickle'))
        final_model = all_models[-1]

        spbn = load(final_model)
        spbn.fit(training_data)

        logl = spbn.slogl(test_data)

        print("Loglik, p " + str(p) + ": " + str(logl))
        print("SHD, p " + str(p) + ": " + str(experiments_helper.shd(spbn, true_model)))
        print("Hamming, p " + str(p) + ": " + str(experiments_helper.hamming(spbn, true_model)))
        print("Type Hamming, p " + str(p) + ": " + str(experiments_helper.hamming_type(spbn)))
        print()

for true_model, name, folders, training_datasets, test_dataset in \
        zip([ecoli70_true, magic_niab_true, magic_irri_true, arth150_true],
            ["ECOLI70", "MAGIC_NIAB", "MAGIC_IRRI", "ARTH150"],
            [
                ['models/ecoli70/200', 'models/ecoli70/2000', 'models/ecoli70/10000'],
                ['models/magic_niab/200', 'models/magic_niab/2000', 'models/magic_niab/10000'],
                ['models/magic_irri/200', 'models/magic_irri/2000', 'models/magic_irri/10000'],
                ['models/arth150/200', 'models/arth150/2000', 'models/arth150/10000']
            ],
            [
                [ecoli70_200, ecoli70_2000, ecoli70_10000],
                [magic_niab_200, magic_niab_2000, magic_niab_10000],
                [magic_irri_200, magic_irri_2000, magic_irri_10000],
                [arth150_200, arth150_2000, arth150_10000]
            ],
            [ecoli70_test, magic_niab_test, magic_irri_test, arth150_test]
           ):

    print(name)
    print("==============")
    print()

    for folder, training_dataset in zip(folders, training_datasets):
        print(folder)
        print()
        compare_models(true_model, folder, training_dataset, test_dataset, experiments_helper.PATIENCE)

    print()