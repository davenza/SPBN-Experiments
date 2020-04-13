import numpy as np
np.random.seed(0)
import pandas as pd
import pathlib
import glob
from pgmpy.models import HybridContinuousModel, LinearGaussianBayesianNetwork
from pgmpy.factors.continuous import NodeType

def shd(estimated, true):
    assert set(estimated.nodes) == set(true.nodes)
    shd_value = 0

    estimated_edges = set(estimated.edges)
    true_edges = set(true.edges)

    for est_edge in estimated_edges:
        if est_edge not in true.edges:
            shd_value += 1
            s, d = est_edge
            if (d, s) in true_edges:
                true_edges.remove((d, s))

    for true_edge in true_edges:
        if true_edge not in estimated_edges:
            shd_value += 1

    return shd_value

def hamming(estimated, true):
    assert set(estimated.nodes) == set(true.nodes)
    hamming_value = 0

    estimated_edges = set(estimated.edges)
    true_edges = set(true.edges)

    for est_edge in estimated_edges:
        if est_edge not in true.edges:
            s, d = est_edge
            if (d, s) in true_edges:
                true_edges.remove((d,s))
            else:
                hamming_value += 1

    for true_edge in true_edges:
        if true_edge not in estimated_edges:
            hamming_value += 1

    return hamming_value

def hamming_type(estimated, true):
    assert set(estimated.nodes) == set(true.nodes)
    hamming_value = 0

    for n in true.nodes:
        if estimated.node_type[n] == NodeType.CKDE:
            hamming_value += 1

    return hamming_value

ecoli70_true = LinearGaussianBayesianNetwork.load_model('ecoli70.pkl')
magic_niab_true = LinearGaussianBayesianNetwork.load_model('magic_niab.pkl')
magic_irri_true = LinearGaussianBayesianNetwork.load_model('magic_irri.pkl')
arth150_true = LinearGaussianBayesianNetwork.load_model('arth150.pkl')


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
    ground_truth_logl = true_model.logpdf_dataset(test_data)

    print("Ground truth loglik: " + str(ground_truth_logl.sum()))
    print("CKDE results:")
    for p in patience:
        ckde_folder = trained_models_folder + '/CKDE/' + str(p)

        all_models = sorted(glob.glob(ckde_folder + '/*.pkl'))
        final_model = all_models[-1]

        hcm = HybridContinuousModel.load_model(final_model)
        hcm.fit(training_data)

        logl = hcm.logpdf_dataset(test_data)

        print("Loglik, p " + str(p) + ": " + str(logl.sum()))
        print("SHD, p " + str(p) + ": " + str(shd(hcm, true_model)))
        print("Hamming, p " + str(p) + ": " + str(hamming(hcm, true_model)))
        print("Type Hamming, p " + str(p) + ": " + str(hamming_type(hcm, true_model)))
        print()

    print("GBN Validation results:")
    for p in patience:
        gbn_val_folder = trained_models_folder + '/GBN_Validation/' + str(p)

        all_models = sorted(glob.glob(gbn_val_folder + '/*.pkl'))
        final_model = all_models[-1]

        hcm = LinearGaussianBayesianNetwork.load_model(final_model)
        hcm.fit(training_data)

        logl = hcm.logpdf_dataset(test_data)

        print("Loglik, p " + str(p) + ": " + str(logl.sum()))
        print("SHD, p " + str(p) + ": " + str(shd(hcm, true_model)))
        print("Hamming, p " + str(p) + ": " + str(hamming(hcm, true_model)))
        print()

    gbn_bic_folder = trained_models_folder + '/GBN_BIC/'
    all_models = sorted(glob.glob(gbn_bic_folder + '/*.pkl'))
    final_model = all_models[-1]

    hcm = LinearGaussianBayesianNetwork.load_model(final_model)
    hcm.fit(training_data)

    logl = hcm.logpdf_dataset(test_data)
    print("GBN BIC results:")
    print("Loglik: " + str(logl.sum()))
    print("SHD: " + str(shd(hcm, true_model)))
    print("Hamming: " + str(hamming(hcm, true_model)))
    print()

    gbn_bge_folder = trained_models_folder + '/GBN_BGe/'
    all_models = sorted(glob.glob(gbn_bge_folder + '/*.pkl'))
    final_model = all_models[-1]

    hcm = LinearGaussianBayesianNetwork.load_model(final_model)
    hcm.fit(training_data)

    logl = hcm.logpdf_dataset(test_data)
    print("GBN BGe results:")
    print("Loglik: " + str(logl.sum()))
    print("SHD: " + str(shd(hcm, true_model)))
    print("Hamming: " + str(hamming(hcm, true_model)))
    print()

patience = [0, 5]
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
        compare_models(true_model, folder, training_dataset, test_dataset, patience)

    print()