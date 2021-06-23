import os
import subprocess
import numpy as np
import pandas as pd
import glob
from pybnesian import load
import matplotlib.pyplot as plt
import experiments_helper
import adjusted_pvalues
import plot_cd_diagram

import tikzplotlib

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

ecoli70_true = load('ecoli70.pickle')
magic_niab_true = load('magic_niab.pickle')
magic_irri_true = load('magic_irri.pickle')
arth150_true = load('arth150.pickle')

# COLOR1 = "#B5EA7F"
# COLOR2 = "#739DF6"
COLOR1 = "#729CF5"
COLOR2 = "#FFB346"
COLOR3 = "#B5EA7F"
COLOR4 = "#00000080"

def extract_info(train_datasets, test_datasets, model_folders, true_models):
    patience = experiments_helper.PATIENCE

    tests = experiments_helper.TESTS

    slogl_true = np.empty((len(train_datasets,)))
    slogl_hc_gbn_bic = np.empty((len(train_datasets), len(train_datasets[0])))
    slogl_hc_gbn_bge = np.empty((len(train_datasets), len(train_datasets[0])))
    slogl_hc_spbn = np.empty((len(train_datasets), len(train_datasets[0]), len(patience)))
    slogl_hc_spbn_ckde = np.empty((len(train_datasets), len(train_datasets[0]), len(patience)))
    slogl_pc_gbn = np.empty((len(train_datasets), len(train_datasets[0]), len(tests)))
    slogl_pc_spbn = np.empty((len(train_datasets), len(train_datasets[0]), len(tests), len(patience)))
    slogl_pc_spbn_ckde = np.empty((len(train_datasets), len(train_datasets[0]), len(tests), len(patience)))

    hmd_hc_gbn_bic = np.empty((len(train_datasets), len(train_datasets[0])))
    hmd_hc_gbn_bge = np.empty((len(train_datasets), len(train_datasets[0])))
    hmd_hc_spbn = np.empty((len(train_datasets), len(train_datasets[0]), len(patience)))
    hmd_hc_spbn_ckde = np.empty((len(train_datasets), len(train_datasets[0]), len(patience)))
    hmd_pc = np.empty((len(train_datasets), len(train_datasets[0]), len(tests)))

    shd_hc_gbn_bic = np.empty((len(train_datasets), len(train_datasets[0])))
    shd_hc_gbn_bge = np.empty((len(train_datasets), len(train_datasets[0])))
    shd_hc_spbn = np.empty((len(train_datasets), len(train_datasets[0]), len(patience)))
    shd_hc_spbn_ckde = np.empty((len(train_datasets), len(train_datasets[0]), len(patience)))
    shd_pc = np.empty((len(train_datasets), len(train_datasets[0]), len(tests)))

    thd_hc_spbn = np.empty((len(train_datasets), len(train_datasets[0]), len(patience)))
    thd_hc_spbn_ckde = np.empty((len(train_datasets), len(train_datasets[0]), len(patience)))
    thd_pc_spbn = np.empty((len(train_datasets), len(train_datasets[0]), len(tests), len(patience)))
    thd_pc_spbn_ckde = np.empty((len(train_datasets), len(train_datasets[0]), len(tests), len(patience)))

    for idx_dataset, (instance_datasets, test_data, dataset_folders, true_model) in enumerate(
            zip(train_datasets, test_datasets, model_folders, true_models)):
        for idx_instances, (training_data, folder) in enumerate(zip(instance_datasets, dataset_folders)):

            slogl_true[idx_dataset] = true_model.slogl(test_data)

            ###########################
            # GBN BIC
            ###########################
            gbn_bic_folder = folder + '/HillClimbing/GBN_BIC/'

            all_models = sorted(glob.glob(gbn_bic_folder + '/*.pickle'))
            final_model = all_models[-1]

            bic = load(final_model)
            bic.fit(training_data)

            slogl_hc_gbn_bic[idx_dataset, idx_instances] = bic.slogl(test_data)
            hmd_hc_gbn_bic[idx_dataset, idx_instances] = experiments_helper.hamming(bic, true_model)
            shd_hc_gbn_bic[idx_dataset, idx_instances] = experiments_helper.shd(bic, true_model)


            ###########################
            # GBN BGe
            ###########################
            gbn_bge_folder = folder + '/HillClimbing/GBN_BGe/'

            all_models = sorted(glob.glob(gbn_bge_folder + '/*.pickle'))
            final_model = all_models[-1]

            bge = load(final_model)
            bge.fit(training_data)

            slogl_hc_gbn_bge[idx_dataset, idx_instances] = bge.slogl(test_data)
            hmd_hc_gbn_bge[idx_dataset, idx_instances] = experiments_helper.hamming(bge, true_model)
            shd_hc_gbn_bge[idx_dataset, idx_instances] = experiments_helper.shd(bge, true_model)

            ###########################
            # HC SPBN
            ###########################
            for idx_p, p in enumerate(patience):
                spbn_hc_folder = folder + '/HillClimbing/SPBN/' + str(p)

                all_models = sorted(glob.glob(spbn_hc_folder + '/*.pickle'))
                final_model = all_models[-1]

                spbn = load(final_model)
                spbn.fit(training_data)

                slogl_hc_spbn[idx_dataset, idx_instances, idx_p] = spbn.slogl(test_data)
                hmd_hc_spbn[idx_dataset, idx_instances, idx_p] = experiments_helper.hamming(spbn, true_model)
                shd_hc_spbn[idx_dataset, idx_instances, idx_p] = experiments_helper.shd(spbn, true_model)
                thd_hc_spbn[idx_dataset, idx_instances, idx_p] = experiments_helper.hamming_type(spbn)

            ###########################
            # HC SPBN CKDE
            ###########################
            for idx_p, p in enumerate(patience):
                spbn_ckde_hc_folder = folder + '/HillClimbing/SPBN_CKDE/' + str(p)

                all_models = sorted(glob.glob(spbn_ckde_hc_folder + '/*.pickle'))
                final_model = all_models[-1]

                spbn_ckde = load(final_model)
                spbn_ckde.fit(training_data)

                slogl_hc_spbn_ckde[idx_dataset, idx_instances, idx_p] = spbn_ckde.slogl(test_data)
                hmd_hc_spbn_ckde[idx_dataset, idx_instances, idx_p] = experiments_helper.hamming(spbn_ckde, true_model)
                shd_hc_spbn_ckde[idx_dataset, idx_instances, idx_p] = experiments_helper.shd(spbn_ckde, true_model)
                thd_hc_spbn_ckde[idx_dataset, idx_instances, idx_p] = experiments_helper.hamming_type(spbn_ckde)

            ###########################
            # PC GBN and PC Graph
            ###########################
            for idx_t, test in enumerate(tests):
                gbn_pc_folder = folder + '/PC/GBN/' + test
                
                all_models = sorted(glob.glob(gbn_pc_folder + '/*.pickle'))
                final_model = all_models[-1]

                gbn_pc = load(final_model)
                gbn_pc.fit(training_data)

                slogl_pc_gbn[idx_dataset, idx_instances, idx_t] = gbn_pc.slogl(test_data)
                hmd_pc[idx_dataset, idx_instances, idx_t] = experiments_helper.hamming(gbn_pc, true_model)
                shd_pc[idx_dataset, idx_instances, idx_t] = experiments_helper.shd(gbn_pc, true_model)

            ###########################
            # PC SPBN
            ###########################
            for idx_t, test in enumerate(tests):
                for idx_p, p in enumerate(patience):
                    spbn_pc_folder = folder + '/PC/SPBN/' + test + '/' + str(p)

                    all_models = sorted(glob.glob(spbn_pc_folder + '/*.pickle'))
                    final_model = all_models[-1]

                    spbn_pc = load(final_model)
                    spbn_pc.fit(training_data)

                    slogl_pc_spbn[idx_dataset, idx_instances, idx_t, idx_p] = spbn_pc.slogl(test_data)
                    thd_pc_spbn[idx_dataset, idx_instances, idx_t, idx_p] = experiments_helper.hamming_type(spbn_pc)

            ###########################
            # PC SPBN CKDE
            ###########################
            for idx_t, test in enumerate(tests):
                for idx_p, p in enumerate(patience):
                    spbn_ckde_pc_folder = folder + '/PC/SPBN_CKDE/' + test + '/' + str(p)

                    all_models = sorted(glob.glob(spbn_ckde_pc_folder + '/*.pickle'))
                    final_model = all_models[-1]

                    spbn_ckde_pc = load(final_model)
                    spbn_ckde_pc.fit(training_data)

                    slogl_pc_spbn_ckde[idx_dataset, idx_instances, idx_t, idx_p] = spbn_ckde_pc.slogl(test_data)
                    thd_pc_spbn_ckde[idx_dataset, idx_instances, idx_t, idx_p] = experiments_helper.hamming_type(spbn_ckde_pc)


    return (slogl_true, slogl_hc_gbn_bic, slogl_hc_gbn_bge, slogl_hc_spbn, slogl_hc_spbn_ckde, slogl_pc_gbn, slogl_pc_spbn, slogl_pc_spbn_ckde), \
           (hmd_hc_gbn_bic, hmd_hc_gbn_bge, hmd_hc_spbn, hmd_hc_spbn_ckde, hmd_pc), \
           (shd_hc_gbn_bic, shd_hc_gbn_bge, shd_hc_spbn, shd_hc_spbn_ckde, shd_pc),\
           (thd_hc_spbn, thd_hc_spbn_ckde, thd_pc_spbn, thd_pc_spbn_ckde)

def plot_table(train_datasets, test_datasets, model_folders, true_models):
    loglikelihood_info, _, _, _ = extract_info(train_datasets, test_datasets, model_folders, true_models)
    (slogl_true, slogl_hc_gbn_bic, slogl_hc_gbn_bge, slogl_hc_spbn,
    slogl_hc_spbn_ckde, slogl_pc_gbn, slogl_pc_spbn, slogl_pc_spbn_ckde) = loglikelihood_info

    for i in range(0, 4, 2):
        line = "GBN BIC"
        for j in range(i, i+2):
            for idx_inst, inst in enumerate(experiments_helper.INSTANCES):
                line += "&$" + str(round(slogl_hc_gbn_bic[j, idx_inst], 2)) + "$"
        line += "\\\\\n"

        line += "GBN BGe"
        for j in range(i, i+2):
            for idx_inst, inst in enumerate(experiments_helper.INSTANCES):
                line += "&$" + str(round(slogl_hc_gbn_bge[j, idx_inst], 2)) + "$"
        line += "\\\\\n"

        line += "HC SPBN-LG"
        for j in range(i, i+2):
            for idx_inst, inst in enumerate(experiments_helper.INSTANCES):
                line += "&$" + str(round(slogl_hc_spbn[j, idx_inst, 1], 2)) + "$"
        line += "\\\\\n"

        line += "HC SPBN-CKDE"
        for j in range(i, i+2):
            for idx_inst, inst in enumerate(experiments_helper.INSTANCES):
                line += "&$" + str(round(slogl_hc_spbn_ckde[j, idx_inst, 1], 2)) + "$"
        line += "\\\\\n"

        line += "PC-PLC GBN"
        for j in range(i, i+2):
            for idx_inst, inst in enumerate(experiments_helper.INSTANCES):
                line += "&$" + str(round(slogl_pc_gbn[j, idx_inst, 0], 2)) + "$"
        line += "\\\\\n"

        line += "PC-RCoT GBN"
        for j in range(i, i+2):
            for idx_inst, inst in enumerate(experiments_helper.INSTANCES):
                line += "&$" + str(round(slogl_pc_gbn[j, idx_inst, 1], 2)) + "$"
        line += "\\\\\n"

        line += "PC-PLC SPBN-LG"
        for j in range(i, i+2):
            for idx_inst, inst in enumerate(experiments_helper.INSTANCES):
                line += "&$" + str(round(slogl_pc_spbn[j, idx_inst, 0, 1], 2)) + "$"
        line += "\\\\\n"

        line += "PC-RCoT SPBN-LG"
        for j in range(i, i+2):
            for idx_inst, inst in enumerate(experiments_helper.INSTANCES):
                line += "&$" + str(round(slogl_pc_spbn[j, idx_inst, 1, 1], 2)) + "$"
        line += "\\\\\n"

        line += "PC-PLC SPBN-CKDE"
        for j in range(i, i+2):
            for idx_inst, inst in enumerate(experiments_helper.INSTANCES):
                line += "&$" + str(round(slogl_pc_spbn_ckde[j, idx_inst, 0, 1], 2)) + "$"
        line += "\\\\\n"

        line += "PC-RCoT SPBN-CKDE"
        for j in range(i, i+2):
            for idx_inst, inst in enumerate(experiments_helper.INSTANCES):
                line += "&$" + str(round(slogl_pc_spbn_ckde[j, idx_inst, 1, 1], 2)) + "$"
        line += "\\\\\n"

        print(line)



def print_apv(avgranks, names, type="bergmann"):
    if type == "bergmann":
        apv = adjusted_pvalues.bergmann_hommel(avgranks, names)
    elif type == "holm":
        apv = adjusted_pvalues.holm(avgranks, names)

    for (pvalue, (alg1, alg2)) in apv:
        print(alg1 + " vs " + alg2  + ": " + str(pvalue))


def plot_cd_diagrams(train_datasets, test_datasets, model_folders, true_models, rename_dict):

    loglikelihood_info, _, _, _ = extract_info(train_datasets, test_datasets, model_folders, true_models)

    (slogl_true, slogl_hc_gbn_bic, slogl_hc_gbn_bge, slogl_hc_spbn,
    slogl_hc_spbn_ckde, slogl_pc_gbn, slogl_pc_spbn, slogl_pc_spbn_ckde) = loglikelihood_info

    datasets = [d + "_" + str(i) for d in experiments_helper.DATASETS for i in experiments_helper.INSTANCES]

    df_algorithms = pd.DataFrame({"Dataset": datasets})
    df_algorithms = df_algorithms.set_index("Dataset")

    df_algorithms["BIC"] = slogl_hc_gbn_bic.reshape(-1)
    df_algorithms["BGe"] = slogl_hc_gbn_bge.reshape(-1)

    # for idx_p, p in enumerate(experiments_helper.PATIENCE):
    df_algorithms["HC_SPBN_5"] = slogl_hc_spbn[:,:,1].reshape(-1)
    df_algorithms["HC_SPBN_CKDE_5"] = slogl_hc_spbn_ckde[:,:,1].reshape(-1)

    for idx_t, test in enumerate(experiments_helper.TESTS):
        df_algorithms["PC_GBN_" + test + "_5"] = slogl_pc_gbn[:,:, idx_t].reshape(-1)
        df_algorithms["PC_SPBN_" + test + "_5"] = slogl_pc_spbn[:,:, idx_t, 1].reshape(-1)
        df_algorithms["PC_SPBN_CKDE_" + test + "_5"] = slogl_pc_spbn_ckde[:,:, idx_t, 1].reshape(-1)

    df_algorithms = df_algorithms.filter(regex="_10000", axis=0)

    rank = df_algorithms.rank(axis=1, ascending=False)
    avgranks = rank.mean().to_numpy()
    names = rank.columns.values

    print_apv(avgranks, names)
    input("Press [Enter]:")

    names = [rename_dict[s] for s in names]

    plot_cd_diagram.graph_ranks(avgranks, names, df_algorithms.shape[0], posthoc_method="cd")
    tikzplotlib.save("plots/Nemenyi_spbn.tex", standalone=True, axis_width="16.5cm", axis_height="5cm")

    plot_cd_diagram.graph_ranks(avgranks, names, df_algorithms.shape[0], posthoc_method="holm")
    tikzplotlib.save("plots/Holm_spbn.tex", standalone=True, axis_width="16.5cm", axis_height="5cm")

    plot_cd_diagram.graph_ranks(avgranks, names, df_algorithms.shape[0], posthoc_method="bergmann")
    tikzplotlib.save("plots/Bergmann_spbn.tex", standalone=True, axis_width="16.5cm", axis_height="5cm")

    os.chdir("plots")
    process = subprocess.Popen('pdflatex Nemenyi_Gaussian.tex'.split())
    process.wait()
    process = subprocess.Popen('pdflatex Holm_Gaussian.tex'.split())
    process.wait()
    process = subprocess.Popen('pdflatex Bergmann_Gaussian.tex'.split())
    process.wait()
    process = subprocess.Popen('evince Bergmann_Gaussian.pdf'.split())
    process.wait()
    os.chdir("..")
    return df_algorithms

def plot_hmd(train_datasets, test_datasets, model_folders, true_models, dataset_names, instance_names):
    tests = experiments_helper.TESTS

    _, hmd_info, _, _ = extract_info(train_datasets, test_datasets, model_folders, true_models)
    hmd_hc_gbn_bic, hmd_hc_gbn_bge, hmd_hc_spbn, hmd_hc_spbn_ckde, hmd_pc = hmd_info

    N = len(train_datasets) * len(train_datasets[0])
    num_bars = len(tests)*2 + 2
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

    t = ax.bar(ind+width*offset, hmd_hc_spbn[:,:, 1].reshape(-1), width, color=COLOR1, linewidth=0.5,
                   align='edge', edgecolor="black")
    offset += 1
    b.append(t)

    t = ax.bar(ind+width*offset, hmd_hc_spbn_ckde[:,:, 1].reshape(-1), width, color=COLOR1, linewidth=0.5,
                   align='edge', edgecolor="black")
    offset += 1
    for bar in t:
        bar.set_hatch('//')
    b.append(t)

    t = ax.bar(ind+width*offset, hmd_pc[:,:, 0].reshape(-1), width, color=COLOR2, linewidth=0.5,
                   align='edge', edgecolor="black")
    offset += 1

    t = ax.bar(ind+width*offset, hmd_pc[:,:, 1].reshape(-1), width, color=COLOR2, linewidth=0.5,
                   align='edge', edgecolor="black")
    offset += 1
    for bar in t:
        bar.set_hatch('//')
    b.append(t)

    t = ax.bar(ind + width * offset, hmd_hc_gbn_bic[:,:].reshape(-1), width, color=COLOR3, linewidth=0.5,
               align='edge', edgecolor="black")
    offset += 1
    b.append(t)

    t = ax.bar(ind + width * offset, hmd_hc_gbn_bge[:,:].reshape(-1), width, color=COLOR4, linewidth=0.5,
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
    tests = experiments_helper.TESTS

    _, _, shd_info, _ = extract_info(train_datasets, test_datasets, model_folders, true_models)
    shd_hc_gbn_bic, shd_hc_gbn_bge, shd_hc_spbn, shd_hc_spbn_ckde, shd_pc = shd_info

    N = len(train_datasets) * len(train_datasets[0])
    num_bars = len(tests)*2 + 2
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

    t = ax.bar(ind+width*offset, shd_hc_spbn[:,:, 1].reshape(-1), width, color=COLOR1, linewidth=0.5,
                   align='edge', edgecolor="black")
    offset += 1
    b.append(t)

    t = ax.bar(ind+width*offset, shd_hc_spbn_ckde[:,:, 1].reshape(-1), width, color=COLOR1, linewidth=0.5,
                   align='edge', edgecolor="black")
    offset += 1
    for bar in t:
        bar.set_hatch('//')
    b.append(t)

    t = ax.bar(ind+width*offset, shd_pc[:,:, 0].reshape(-1), width, color=COLOR2, linewidth=0.5,
                   align='edge', edgecolor="black")
    offset += 1

    t = ax.bar(ind+width*offset, shd_pc[:,:, 1].reshape(-1), width, color=COLOR2, linewidth=0.5,
                   align='edge', edgecolor="black")
    offset += 1
    for bar in t:
        bar.set_hatch('//')
    b.append(t)

    t = ax.bar(ind + width * offset, shd_hc_gbn_bic[:,:].reshape(-1), width, color=COLOR3, linewidth=0.5,
               align='edge', edgecolor="black")
    offset += 1
    b.append(t)

    t = ax.bar(ind + width * offset, shd_hc_gbn_bge[:,:].reshape(-1), width, color=COLOR4, linewidth=0.5,
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
    tests = experiments_helper.TESTS

    _, _, _, thd_info = extract_info(train_datasets, test_datasets, model_folders, true_models)
    thd_hc_spbn, thd_hc_spbn_ckde, thd_pc_spbn, thd_pc_spbn_ckde = thd_info

    N = len(train_datasets) * len(train_datasets[0])
    num_bars = 6
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

    HATCH_TYPE = '-'
    import matplotlib as mpl
    mpl.rcParams['hatch.linewidth'] = 5  # previous pdf hatch linewidth

    t = ax.bar(ind+width*offset, thd_hc_spbn[:,:, 1].reshape(-1), width, color=COLOR1, linewidth=0.5,
                   align='edge', edgecolor="black")
    offset += 1
    b.append(t)

    t = ax.bar(ind+width*offset, thd_hc_spbn_ckde[:,:, 1].reshape(-1), width, color=COLOR1, linewidth=0.5,
                   align='edge', edgecolor="black")
    offset += 1
    for bar in t:
        bar.set_hatch(HATCH_TYPE)
    b.append(t)

    t = ax.bar(ind+width*offset, thd_pc_spbn[:,:,0,1].reshape(-1), width, color=COLOR2, linewidth=0.5,
                   align='edge', edgecolor="black")
    offset += 1
    b.append(t)

    t = ax.bar(ind+width*offset, thd_pc_spbn_ckde[:,:,0,1].reshape(-1), width, color=COLOR2, linewidth=0.5,
                   align='edge', edgecolor="black")
    offset += 1
    for bar in t:
        bar.set_hatch(HATCH_TYPE)
    b.append(t)

    t = ax.bar(ind+width*offset, thd_pc_spbn[:,:,1,1].reshape(-1), width, color=COLOR3, linewidth=0.5,
                   align='edge', edgecolor="black")
    offset += 1
    b.append(t)

    t = ax.bar(ind+width*offset, thd_pc_spbn_ckde[:,:,1,1].reshape(-1), width, color=COLOR3, linewidth=0.5,
                   align='edge', edgecolor="black")
    offset += 1
    for bar in t:
        bar.set_hatch(HATCH_TYPE)
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

    rename_dict = {
        'BIC': r'GBN BIC',
        'BGe': r'GBN BGe',
        'HC_SPBN_5': r'HC SPBN-LG',
        'HC_SPBN_CKDE_5': r'HC SPBN-CKDE',
        'PC_GBN_LinearCorrelation_5': r'PC-PLC GBN',
        'PC_GBN_RCoT_5': r'PC-RCoT GBN',
        'PC_SPBN_LinearCorrelation_5': r'PC-PLC SPBN-LG',
        'PC_SPBN_RCoT_5': r'PC-RCoT SPBN-LG',
        'PC_SPBN_CKDE_LinearCorrelation_5': r'PC-PLC SPBN-CKDE',
        'PC_SPBN_CKDE_RCoT_5': r'PC-RCoT SPBN-CKDE',
    }

    # plot_table(train_datasets, test_datasets, model_folders, true_models,)
    # plot_cd_diagrams(train_datasets, test_datasets, model_folders, true_models, rename_dict)
    plot_hmd(train_datasets, test_datasets, model_folders, true_models, dataset_names, instance_names)
    plot_shd(train_datasets, test_datasets, model_folders, true_models, dataset_names, instance_names)
    plot_thd(train_datasets, test_datasets, model_folders, true_models, dataset_names, instance_names)