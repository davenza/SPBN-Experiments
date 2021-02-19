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

    print("GBN Validation results:")
    for p in patience:
        gbn_val_folder = trained_models_folder + '/HillClimbing/GBN_Validation/' + str(p)

        all_models = sorted(glob.glob(gbn_val_folder + '/*.pickle'))
        final_model = all_models[-1]

        gbn = load(final_model)
        gbn.fit(training_data)

        slogl = gbn.slogl(test_data)

        print("Loglik, p " + str(p) + ": " + str(slogl))
        print("SHD, p " + str(p) + ": " + str(experiments_helper.shd(gbn, true_model)))
        print("Hamming, p " + str(p) + ": " + str(experiments_helper.hamming(gbn, true_model)))
        print()

    gbn_bic_folder = trained_models_folder + '/HillClimbing/GBN_BIC/'
    all_models = sorted(glob.glob(gbn_bic_folder + '/*.pickle'))
    final_model = all_models[-1]

    gbn = load(final_model)
    gbn.fit(training_data)

    slogl = gbn.slogl(test_data)
    print("GBN BIC results:")
    print("Loglik: " + str(slogl))
    print("SHD: " + str(experiments_helper.shd(gbn, true_model)))
    print("Hamming: " + str(experiments_helper.hamming(gbn, true_model)))
    print()

    gbn_bge_folder = trained_models_folder + '/HillClimbing/GBN_BGe/'
    all_models = sorted(glob.glob(gbn_bge_folder + '/*.pickle'))
    final_model = all_models[-1]

    gbn = load(final_model)
    gbn.fit(training_data)

    slogl = gbn.slogl(test_data)
    print("GBN BGe results:")
    print("Loglik: " + str(slogl)
    print("SHD: " + str(experiments_helper.shd(gbn, true_model)))
    print("Hamming: " + str(experiments_helper.hamming(gbn, true_model)))
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