import numpy as np
import pandas as pd
import glob

import matplotlib.pyplot as plt
from pgmpy.models import HybridContinuousModel, LinearGaussianBayesianNetwork
from pgmpy.factors.continuous import NodeType

import tikzplotlib

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

ecoli70_true = LinearGaussianBayesianNetwork.load_model('ecoli70.pkl')
magic_niab_true = LinearGaussianBayesianNetwork.load_model('magic_niab.pkl')
magic_irri_true = LinearGaussianBayesianNetwork.load_model('magic_irri.pkl')
arth150_true = LinearGaussianBayesianNetwork.load_model('arth150.pkl')

# COLOR1 = "#B5EA7F"
# COLOR2 = "#739DF6"
COLOR1 = "#729CF5"
COLOR2 = "#FFB346"
COLOR3 = "#B5EA7F"
COLOR4 = "#00000080"

def extract_info(train_datasets, test_datasets, model_folders, true_models):
    patience = [0, 5]

    logl_true = np.empty((len(train_datasets,)))
    logl_ckde = np.empty((len(train_datasets), len(train_datasets[0]), len(patience)))
    logl_gbn_val = np.empty((len(train_datasets), len(train_datasets[0]), len(patience)))
    logl_gbn_bic = np.empty((len(train_datasets), len(train_datasets[0])))
    logl_gbn_bge = np.empty((len(train_datasets), len(train_datasets[0])))

    hmd_ckde = np.empty((len(train_datasets), len(train_datasets[0]), len(patience)))
    hmd_gbn_val = np.empty((len(train_datasets), len(train_datasets[0]), len(patience)))
    hmd_gbn_bic = np.empty((len(train_datasets), len(train_datasets[0])))
    hmd_gbn_bge = np.empty((len(train_datasets), len(train_datasets[0])))

    shd_ckde = np.empty((len(train_datasets), len(train_datasets[0]), len(patience)))
    shd_gbn_val = np.empty((len(train_datasets), len(train_datasets[0]), len(patience)))
    shd_gbn_bic = np.empty((len(train_datasets), len(train_datasets[0])))
    shd_gbn_bge = np.empty((len(train_datasets), len(train_datasets[0])))

    thd_ckde = np.empty((len(train_datasets), len(train_datasets[0]), len(patience)))

    for idx_dataset, (instance_datasets, test_data, dataset_folders, true_model) in enumerate(
            zip(train_datasets, test_datasets, model_folders, true_models)):
        for idx_instances, (training_data, folder) in enumerate(zip(instance_datasets, dataset_folders)):

            logl_true[idx_dataset] = true_model.logpdf_dataset(test_data).sum()
            for idx_p, p in enumerate(patience):
                ckde_folder = folder + '/CKDE/' + str(p)

                all_models = sorted(glob.glob(ckde_folder + '/*.pkl'))
                final_model = all_models[-1]

                hcm = HybridContinuousModel.load_model(final_model)
                hcm.fit(training_data)

                logl_ckde[idx_dataset, idx_instances, idx_p] = hcm.logpdf_dataset(test_data).sum()
                hmd_ckde[idx_dataset, idx_instances, idx_p] = hamming(hcm, true_model)
                shd_ckde[idx_dataset, idx_instances, idx_p] = shd(hcm, true_model)
                thd_ckde[idx_dataset, idx_instances, idx_p] = hamming_type(hcm, true_model)

            for idx_p, p in enumerate(patience):
                gbn_val_folder = folder + '/GBN_Validation/' + str(p)

                all_models = sorted(glob.glob(gbn_val_folder + '/*.pkl'))
                final_model = all_models[-1]

                lbn = LinearGaussianBayesianNetwork.load_model(final_model)
                lbn.fit(training_data)

                logl_gbn_val[idx_dataset, idx_instances, idx_p] = lbn.logpdf_dataset(test_data).sum()
                hmd_gbn_val[idx_dataset, idx_instances, idx_p] = hamming(lbn, true_model)
                shd_gbn_val[idx_dataset, idx_instances, idx_p] = shd(lbn, true_model)

            gbn_bic_folder = folder + '/GBN_BIC/'

            all_models = sorted(glob.glob(gbn_bic_folder + '/*.pkl'))
            final_model = all_models[-1]

            lbn = LinearGaussianBayesianNetwork.load_model(final_model)
            lbn.fit(training_data)

            logl_gbn_bic[idx_dataset, idx_instances] = lbn.logpdf_dataset(test_data).sum()
            hmd_gbn_bic[idx_dataset, idx_instances] = hamming(lbn, true_model)
            shd_gbn_bic[idx_dataset, idx_instances] = shd(lbn, true_model)

            gbn_bge_folder = folder + '/GBN_BGe/'

            all_models = sorted(glob.glob(gbn_bge_folder + '/*.pkl'))
            final_model = all_models[-1]

            lbn = LinearGaussianBayesianNetwork.load_model(final_model)
            lbn.fit(training_data)

            logl_gbn_bge[idx_dataset, idx_instances] = lbn.logpdf_dataset(test_data).sum()
            hmd_gbn_bge[idx_dataset, idx_instances] = hamming(lbn, true_model)
            shd_gbn_bge[idx_dataset, idx_instances] = shd(lbn, true_model)

    return (logl_true, logl_ckde, logl_gbn_val, logl_gbn_bic, logl_gbn_bge), \
           (hmd_ckde, hmd_gbn_val, hmd_gbn_bic, hmd_gbn_bge), \
           (shd_ckde, shd_gbn_val, shd_gbn_bic, shd_gbn_bge),\
           thd_ckde


def plot_likelihood(train_datasets, test_datasets, model_folders, true_models):
    patience = [0, 5]

    loglikelihood_info, _, _, _ = extract_info(train_datasets, test_datasets, model_folders, true_models)
    logl_true, logl_ckde, logl_gbn_val, logl_gbn_bic, logl_gbn_bge = loglikelihood_info

    N = len(train_datasets) * len(train_datasets[0])
    ind = np.arange(N)
    b = []

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for idx_p, p in enumerate(patience):
        t = ax.plot(ind, logl_ckde[:,:, idx_p].reshape(-1))
        b.append(t)

    for idx_p, p in enumerate(patience):
        t = ax.plot(ind, logl_gbn_val[:, :, idx_p].reshape(-1))
        b.append(t)

    t = ax.plot(ind, logl_gbn_bic[:, :].reshape(-1))
    b.append(t)

    t = ax.plot(ind, logl_gbn_bge[:, :].reshape(-1))
    b.append(t)
    plt.show()

def plot_hmd(train_datasets, test_datasets, model_folders, true_models, dataset_names, instance_names):
    patience = [0, 5]

    _, hmd_info, _, _ = extract_info(train_datasets, test_datasets, model_folders, true_models)
    hmd_ckde, hmd_gbn_val, hmd_gbn_bic, hmd_gbn_bge = hmd_info

    N = len(train_datasets) * len(train_datasets[0])
    num_bars = len(patience)*2 + 2
    ind = np.arange(N, dtype=np.float64)

    # Between dataset groups separation
    n_instances = len(train_datasets[0])
    for i in range(1, len(train_datasets)):
        ind[n_instances*i:] += 0.25

    width = (1 - 0.1) / num_bars

    fig = plt.figure()
    ax = fig.add_subplot(111)

    b = []

    offset = 0

    for idx_p, p in enumerate(patience):
        t = ax.bar(ind+width*offset, hmd_ckde[:,:, idx_p].reshape(-1), width, color=COLOR1, linewidth=0.5,
                   align='edge', edgecolor="black")
        offset += 1
        if p == 5:
            for bar in t:
                bar.set_hatch('//')
        b.append(t)

    for idx_p, p in enumerate(patience):
        t = ax.bar(ind+width*offset, hmd_gbn_val[:,:, idx_p].reshape(-1), width, color=COLOR2, linewidth=0.5,
                   align='edge', edgecolor="black")
        offset += 1
        if p == 5:
            for bar in t:
                bar.set_hatch('//')
        b.append(t)

    t = ax.bar(ind + width * offset, hmd_gbn_bic[:,:].reshape(-1), width, color=COLOR3, linewidth=0.5,
               align='edge', edgecolor="black")
    offset += 1
    b.append(t)

    t = ax.bar(ind + width * offset, hmd_gbn_bge[:,:].reshape(-1), width, color=COLOR4, linewidth=0.5,
               align='edge', edgecolor="black")
    offset += 1
    b.append(t)

    ax.set_ylabel('Hamming distance')
    ax.set_xticks(ind + (1 - 0.1) / 2)
    labels = instance_names * len(train_datasets)
    ax.set_xticklabels(labels)
    # ax.tick_params(axis='x', rotation=90)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax1_ticks = ind + (1 - 0.1) / 2
    ax2.set_xticks(ax1_ticks[1::3])
    ax2.set_xticklabels(dataset_names)
    ax2.xaxis.set_ticks_position('bottom')

    for t in ax2.get_xticklabels():
        t.set_y(-0.04)

    ax2.tick_params(axis='both', which='both', length=0)
    tikzplotlib.save("plots/hmd.tex", standalone=True, axis_width="25cm", axis_height="10cm")

def plot_shd(train_datasets, test_datasets, model_folders, true_models, dataset_names, instance_names):
    patience = [0, 5]

    _, _, shd_info, _ = extract_info(train_datasets, test_datasets, model_folders, true_models)
    shd_ckde, shd_gbn_val, shd_gbn_bic, shd_gbn_bge = shd_info

    N = len(train_datasets) * len(train_datasets[0])
    num_bars = len(patience)*2 + 2
    ind = np.arange(N, dtype=np.float64)

    # Between dataset groups separation
    n_instances = len(train_datasets[0])
    for i in range(1, len(train_datasets)):
        ind[n_instances*i:] += 0.25

    width = (1 - 0.1) / num_bars

    fig = plt.figure()
    ax = fig.add_subplot(111)

    b = []

    offset = 0

    for idx_p, p in enumerate(patience):
        t = ax.bar(ind+width*offset, shd_ckde[:,:, idx_p].reshape(-1), width, color=COLOR1, linewidth=0.5,
                   align='edge', edgecolor="black")
        offset += 1
        if p == 5:
            for bar in t:
                bar.set_hatch('//')
        b.append(t)

    for idx_p, p in enumerate(patience):
        t = ax.bar(ind+width*offset, shd_gbn_val[:,:, idx_p].reshape(-1), width, color=COLOR2, linewidth=0.5,
                   align='edge', edgecolor="black")
        offset += 1
        if p == 5:
            for bar in t:
                bar.set_hatch('//')
        b.append(t)

    t = ax.bar(ind + width * offset, shd_gbn_bic[:,:].reshape(-1), width, color=COLOR3, linewidth=0.5,
               align='edge', edgecolor="black")
    offset += 1
    b.append(t)

    t = ax.bar(ind + width * offset, shd_gbn_bge[:,:].reshape(-1), width, color=COLOR4, linewidth=0.5,
               align='edge', edgecolor="black")
    offset += 1
    b.append(t)

    ax.set_ylabel('Structural Hamming distance')
    ax.set_xticks(ind + (1 - 0.1) / 2)
    labels = instance_names * len(train_datasets)
    ax.set_xticklabels(labels)
    # ax.tick_params(axis='x', rotation=90)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax1_ticks = ind + (1 - 0.1) / 2
    ax2.set_xticks(ax1_ticks[1::3])
    ax2.set_xticklabels(dataset_names)
    ax2.xaxis.set_ticks_position('bottom')

    for t in ax2.get_xticklabels():
        t.set_y(-0.04)

    ax2.tick_params(axis='both', which='both', length=0)
    tikzplotlib.save("plots/shd.tex", standalone=True, axis_width="25cm", axis_height="10cm")

def plot_thd(train_datasets, test_datasets, model_folders, true_models, dataset_names, instance_names):
    patience = [5]

    _, _, _, thd_ckde = extract_info(train_datasets, test_datasets, model_folders, true_models)
    # shd_ckde, shd_gbn_val, shd_gbn_bic, shd_gbn_bge = shd_info

    N = len(train_datasets) * len(train_datasets[0])
    num_bars = len(patience)
    ind = np.arange(N, dtype=np.float64)

    # Between dataset groups separation
    n_instances = len(train_datasets[0])
    for i in range(1, len(train_datasets)):
        ind[n_instances*i:] += 0.25

    width = (1 - 0.1) / num_bars

    fig = plt.figure()
    ax = fig.add_subplot(111)

    b = []

    offset = 0

    # for idx_p, p in enumerate(patience):
    # print(ind+width*offset)
    t = ax.bar(ind+width*offset, thd_ckde[:,:,1].reshape(-1), width, color=COLOR1, linewidth=0.5,
               align='edge', edgecolor="black")
    offset += 1
    for bar in t:
        bar.set_hatch('//')
    b.append(t)

    ax.set_ylabel('Node type Hamming distance')
    ax.set_xticks(ind + (1 - 0.1) / 2)
    labels = instance_names * len(train_datasets)
    ax.set_xticklabels(labels)
    # ax.tick_params(axis='x', rotation=90)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax1_ticks = ind + (1 - 0.1) / 2
    ax2.set_xticks(ax1_ticks[1::3])
    ax2.set_xticklabels(dataset_names)
    ax2.xaxis.set_ticks_position('bottom')

    for t in ax2.get_xticklabels():
        t.set_y(-0.04)

    ax2.tick_params(axis='both', which='both', length=0)
    # plt.show()
    tikzplotlib.save("plots/thd.tex", standalone=True, axis_width="25cm", axis_height="10cm")


if __name__ == '__main__':
    train_datasets = [
        [ecoli70_200, ecoli70_2000, ecoli70_10000],
        [magic_niab_200, magic_niab_2000, magic_niab_10000],
        [magic_irri_200, magic_irri_2000, magic_irri_10000],
        [arth150_200, arth150_2000, arth150_10000]
    ]

    test_datasets = [ecoli70_test, magic_niab_test, magic_irri_test, arth150_test]

    model_folders = [
        ['models/ecoli70/200', 'models/ecoli70/2000', 'models/ecoli70/10000'],
        ['models/magic_niab/200', 'models/magic_niab/2000', 'models/magic_niab/10000'],
        ['models/magic_irri/200', 'models/magic_irri/2000', 'models/magic_irri/10000'],
        ['models/arth150/200', 'models/arth150/2000', 'models/arth150/10000'],
    ]

    true_models = [ecoli70_true, magic_niab_true, magic_irri_true, arth150_true]

    dataset_names = ["ECOLI70", "MAGIC-NIAB", "MAGIC-IRRI", "ARTH150"]
    instance_names = ["200", "2000", "10000"]

    # plot_likelihood(train_datasets, test_datasets, model_folders, true_models)
    # plot_hmd(train_datasets, test_datasets, model_folders, true_models, dataset_names, instance_names)
    # plot_shd(train_datasets, test_datasets, model_folders, true_models, dataset_names, instance_names)
    plot_thd(train_datasets, test_datasets, model_folders, true_models, dataset_names, instance_names)