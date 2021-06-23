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

def extract_info(model_folders, true_models):
    patience = experiments_helper.PATIENCE

    tests = experiments_helper.TESTS
    simulations = experiments_helper.NUM_SIMULATIONS
    instances = experiments_helper.INSTANCES

    num_datasets = len(experiments_helper.DATASETS)

    slogl_true = np.empty((num_datasets, simulations))
    slogl_hc_gbn_bic = np.empty((num_datasets, len(instances), simulations))
    slogl_hc_gbn_bge = np.empty((num_datasets, len(instances), simulations))
    slogl_hc_spbn = np.empty((num_datasets, len(instances), len(patience), simulations))
    slogl_hc_spbn_ckde = np.empty((num_datasets, len(instances), len(patience), simulations))
    slogl_pc_gbn = np.empty((num_datasets, len(instances), len(tests), simulations))
    slogl_pc_spbn = np.empty((num_datasets, len(instances), len(tests), len(patience), simulations))
    slogl_pc_spbn_ckde = np.empty((num_datasets, len(instances), len(tests), len(patience), simulations))

    hmd_hc_gbn_bic = np.empty((num_datasets, len(instances), simulations))
    hmd_hc_gbn_bge = np.empty((num_datasets, len(instances), simulations))
    hmd_hc_spbn = np.empty((num_datasets, len(instances), len(patience), simulations))
    hmd_hc_spbn_ckde = np.empty((num_datasets, len(instances), len(patience), simulations))
    hmd_pc = np.empty((num_datasets, len(instances), len(tests), simulations))

    shd_hc_gbn_bic = np.empty((num_datasets, len(instances), simulations))
    shd_hc_gbn_bge = np.empty((num_datasets, len(instances), simulations))
    shd_hc_spbn = np.empty((num_datasets, len(instances), len(patience), simulations))
    shd_hc_spbn_ckde = np.empty((num_datasets, len(instances), len(patience), simulations))
    shd_pc = np.empty((num_datasets, len(instances), len(tests), simulations))

    thd_hc_spbn = np.empty((num_datasets, len(instances), len(patience), simulations))
    thd_hc_spbn_ckde = np.empty((num_datasets, len(instances), len(patience), simulations))
    thd_pc_spbn = np.empty((num_datasets, len(instances), len(tests), len(patience), simulations))
    thd_pc_spbn_ckde = np.empty((num_datasets, len(instances), len(tests), len(patience), simulations))

    for idx_dataset, (model_folder, dataset_prefix, true_model) in enumerate(zip(model_folders, experiments_helper.DATASETS, true_models)):
        for idx_instances, i in enumerate(experiments_helper.INSTANCES):
            for s in range(experiments_helper.NUM_SIMULATIONS):
                train_data_name = 'data/' + dataset_prefix + "_" + str(s).zfill(3) + '_' + str(i) + '.csv' 
                test_data_name = 'data/' + dataset_prefix + "_" + str(s).zfill(3) + '_test.csv' 

                train_data = pd.read_csv(train_data_name)
                test_data = pd.read_csv(test_data_name)

                slogl_true[idx_dataset, s] = true_model.slogl(test_data)

                ###########################
                # GBN BIC
                ###########################
                gbn_bic_folder = model_folder + '/' + str(s).zfill(3) + '/' + str(i) + '/HillClimbing/GBN_BIC/'

                all_models = sorted(glob.glob(gbn_bic_folder + '/*.pickle'))
                final_model = all_models[-1]

                bic = load(final_model)
                bic.fit(train_data)

                slogl_hc_gbn_bic[idx_dataset, idx_instances, s] = bic.slogl(test_data)
                hmd_hc_gbn_bic[idx_dataset, idx_instances, s] = experiments_helper.hamming(bic, true_model)
                shd_hc_gbn_bic[idx_dataset, idx_instances, s] = experiments_helper.shd(bic, true_model)


                ###########################
                # GBN BGe
                ###########################
                gbn_bge_folder = model_folder + '/' + str(s).zfill(3) + '/' + str(i) + '/HillClimbing/GBN_BGe/'

                all_models = sorted(glob.glob(gbn_bge_folder + '/*.pickle'))
                final_model = all_models[-1]

                bge = load(final_model)
                bge.fit(train_data)

                slogl_hc_gbn_bge[idx_dataset, idx_instances, s] = bge.slogl(test_data)
                hmd_hc_gbn_bge[idx_dataset, idx_instances, s] = experiments_helper.hamming(bge, true_model)
                shd_hc_gbn_bge[idx_dataset, idx_instances, s] = experiments_helper.shd(bge, true_model)

                ###########################
                # HC SPBN
                ###########################
                for idx_p, p in enumerate(patience):
                    spbn_hc_folder = model_folder + '/' + str(s).zfill(3) + '/' + str(i) + '/HillClimbing/SPBN/' + str(p)

                    all_models = sorted(glob.glob(spbn_hc_folder + '/*.pickle'))
                    final_model = all_models[-1]

                    spbn = load(final_model)
                    spbn.fit(train_data)

                    slogl_hc_spbn[idx_dataset, idx_instances, idx_p, s] = spbn.slogl(test_data)
                    hmd_hc_spbn[idx_dataset, idx_instances, idx_p, s] = experiments_helper.hamming(spbn, true_model)
                    shd_hc_spbn[idx_dataset, idx_instances, idx_p, s] = experiments_helper.shd(spbn, true_model)
                    thd_hc_spbn[idx_dataset, idx_instances, idx_p, s] = experiments_helper.hamming_type(spbn)

                ###########################
                # HC SPBN CKDE
                ###########################
                for idx_p, p in enumerate(patience):
                    spbn_ckde_hc_folder = model_folder + '/' + str(s).zfill(3) + '/' + str(i) + '/HillClimbing/SPBN_CKDE/' + str(p)

                    all_models = sorted(glob.glob(spbn_ckde_hc_folder + '/*.pickle'))
                    final_model = all_models[-1]

                    spbn_ckde = load(final_model)
                    spbn_ckde.fit(train_data)

                    slogl_hc_spbn_ckde[idx_dataset, idx_instances, idx_p, s] = spbn_ckde.slogl(test_data)
                    hmd_hc_spbn_ckde[idx_dataset, idx_instances, idx_p, s] = experiments_helper.hamming(spbn_ckde, true_model)
                    shd_hc_spbn_ckde[idx_dataset, idx_instances, idx_p, s] = experiments_helper.shd(spbn_ckde, true_model)
                    thd_hc_spbn_ckde[idx_dataset, idx_instances, idx_p, s] = experiments_helper.hamming_type(spbn_ckde)

                ###########################
                # PC GBN and PC Graph
                ###########################
                for idx_t, test in enumerate(tests):
                    gbn_pc_folder = model_folder + '/' + str(s).zfill(3) + '/' + str(i) + '/PC/GBN/' + test
                    
                    all_models = sorted(glob.glob(gbn_pc_folder + '/*.pickle'))
                    final_model = all_models[-1]

                    gbn_pc = load(final_model)
                    gbn_pc.fit(train_data)

                    slogl_pc_gbn[idx_dataset, idx_instances, idx_t, s] = gbn_pc.slogl(test_data)
                    hmd_pc[idx_dataset, idx_instances, idx_t, s] = experiments_helper.hamming(gbn_pc, true_model)
                    shd_pc[idx_dataset, idx_instances, idx_t, s] = experiments_helper.shd(gbn_pc, true_model)

                ###########################
                # PC SPBN
                ###########################
                for idx_t, test in enumerate(tests):
                    for idx_p, p in enumerate(patience):
                        spbn_pc_folder = model_folder + '/' + str(s).zfill(3) + '/' + str(i) + '/PC/SPBN/' + test + '/' + str(p)

                        all_models = sorted(glob.glob(spbn_pc_folder + '/*.pickle'))
                        final_model = all_models[-1]

                        spbn_pc = load(final_model)
                        spbn_pc.fit(train_data)

                        slogl_pc_spbn[idx_dataset, idx_instances, idx_t, idx_p, s] = spbn_pc.slogl(test_data)
                        thd_pc_spbn[idx_dataset, idx_instances, idx_t, idx_p, s] = experiments_helper.hamming_type(spbn_pc)

                ###########################
                # PC SPBN CKDE
                ###########################
                for idx_t, test in enumerate(tests):
                    for idx_p, p in enumerate(patience):
                        spbn_ckde_pc_folder = model_folder + '/' + str(s).zfill(3) + '/' + str(i) + '/PC/SPBN_CKDE/' + test + '/' + str(p)

                        all_models = sorted(glob.glob(spbn_ckde_pc_folder + '/*.pickle'))
                        final_model = all_models[-1]

                        spbn_ckde_pc = load(final_model)
                        spbn_ckde_pc.fit(train_data)

                        slogl_pc_spbn_ckde[idx_dataset, idx_instances, idx_t, idx_p, s] = spbn_ckde_pc.slogl(test_data)
                        thd_pc_spbn_ckde[idx_dataset, idx_instances, idx_t, idx_p, s] = experiments_helper.hamming_type(spbn_ckde_pc)


    return (slogl_true.mean(axis=-1), slogl_hc_gbn_bic.mean(axis=-1), slogl_hc_gbn_bge.mean(axis=-1), slogl_hc_spbn.mean(axis=-1), 
            slogl_hc_spbn_ckde.mean(axis=-1), slogl_pc_gbn.mean(axis=-1), slogl_pc_spbn.mean(axis=-1), slogl_pc_spbn_ckde.mean(axis=-1)), \
           (hmd_hc_gbn_bic.mean(axis=-1), hmd_hc_gbn_bge.mean(axis=-1), hmd_hc_spbn.mean(axis=-1), hmd_hc_spbn_ckde.mean(axis=-1), hmd_pc.mean(axis=-1)), \
           (shd_hc_gbn_bic.mean(axis=-1), shd_hc_gbn_bge.mean(axis=-1), shd_hc_spbn.mean(axis=-1), shd_hc_spbn_ckde.mean(axis=-1), shd_pc.mean(axis=-1)),\
           (thd_hc_spbn.mean(axis=-1), thd_hc_spbn_ckde.mean(axis=-1), thd_pc_spbn.mean(axis=-1), thd_pc_spbn_ckde.mean(axis=-1))

def plot_table(model_folders, true_models):
    loglikelihood_info, _, _, _ = extract_info(model_folders, true_models)
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


def plot_cd_diagrams(model_folders, true_models, rename_dict):
    loglikelihood_info, _, _, _ = extract_info(model_folders, true_models)

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

def plot_hmd(model_folders, true_models, dataset_names, instance_names):
    tests = experiments_helper.TESTS

    _, hmd_info, _, _ = extract_info(model_folders, true_models)
    hmd_hc_gbn_bic, hmd_hc_gbn_bge, hmd_hc_spbn, hmd_hc_spbn_ckde, hmd_pc = hmd_info

    N = len(experiments_helper.DATASETS) * len(experiments_helper.INSTANCES)
    num_bars = len(tests)*2 + 2
    ind = np.arange(N, dtype=np.float64)

    # Between dataset groups separation
    n_instances = len(experiments_helper.INSTANCES)
    for i in range(1, len(experiments_helper.DATASETS)):
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
    labels = instance_names * len(experiments_helper.DATASETS)
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

def plot_shd(model_folders, true_models, dataset_names, instance_names):
    tests = experiments_helper.TESTS

    _, _, shd_info, _ = extract_info(model_folders, true_models)
    shd_hc_gbn_bic, shd_hc_gbn_bge, shd_hc_spbn, shd_hc_spbn_ckde, shd_pc = shd_info

    N = len(experiments_helper.DATASETS) * len(experiments_helper.INSTANCES)
    num_bars = len(tests)*2 + 2
    ind = np.arange(N, dtype=np.float64)

    # Between dataset groups separation
    n_instances = len(experiments_helper.INSTANCES)
    for i in range(1, len(experiments_helper.DATASETS)):
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
    labels = instance_names * len(experiments_helper.DATASETS)
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

def plot_thd(model_folders, true_models, dataset_names, instance_names):
    tests = experiments_helper.TESTS

    _, _, _, thd_info = extract_info(model_folders, true_models)
    thd_hc_spbn, thd_hc_spbn_ckde, thd_pc_spbn, thd_pc_spbn_ckde = thd_info

    N = len(experiments_helper.DATASETS) * len(experiments_helper.INSTANCES)
    num_bars = 6
    ind = np.arange(N, dtype=np.float64)

    # Between dataset groups separation
    n_instances = len(experiments_helper.INSTANCES)
    for i in range(1, len(experiments_helper.DATASETS)):
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
    labels = instance_names * len(experiments_helper.DATASETS)
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
    model_folders = ['models/ecoli70', 'models/magic_niab', 'models/magic_irri', 'models/arth150']

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

    plot_table(model_folders, true_models)
    # plot_cd_diagrams(model_folders, true_models, rename_dict)
    # plot_hmd(model_folders, true_models, dataset_names, instance_names)
    # plot_shd(model_folders, true_models, dataset_names, instance_names)
    # plot_thd(model_folders, true_models, dataset_names, instance_names)