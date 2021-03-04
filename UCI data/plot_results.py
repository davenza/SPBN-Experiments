import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_cd_diagram

import subprocess
import glob

import os
import experiments_helper

from pybnesian import load
from pybnesian.factors import NodeType

import tikzplotlib
import adjusted_pvalues
import experiments_helper

def print_apv(avgranks, names, type="bergmann"):

    if type == "bergmann":
        apv = adjusted_pvalues.bergmann_hommel(avgranks, names)
    elif type == "holm":
        apv = adjusted_pvalues.holm(avgranks, names)

    for (pvalue, (alg1, alg2)) in apv:
        print(alg1 + " vs " + alg2  + ": " + str(pvalue))

def num_instances_df():
    files = experiments_helper.find_crossvalidation_datasets()
    names = []
    num_instances = []
    for file in files:
        x = experiments_helper.validate_dataset(file, experiments_helper.TRAINING_FOLDS)
        if x is not None:
            dataset, result_folder = x

            names.append(os.path.basename(result_folder))
            num_instances.append(dataset.shape[0])

    return pd.DataFrame({"Dataset": names, "N": num_instances})


def plot_cd_diagrams(rename_dict):
    df_num_instances = num_instances_df()
    df_num_instances = df_num_instances.set_index("Dataset")

    df_hc_gbn = pd.read_csv('results_hc_gbn.csv')
    df_hc_kdebn = pd.read_csv('results_hc_kdebn.csv')
    df_hc_spbn = pd.read_csv('results_hc_spbn.csv')
    df_hc_spbn_ckde = pd.read_csv('results_hc_spbn_ckde.csv')

    df_pc_gbn = pd.read_csv('results_pc_gbn.csv')
    df_pc_kdebn = pd.read_csv('results_pc_kdebn.csv')
    df_pc_spbn = pd.read_csv('results_pc_spbn.csv')
    df_pc_spbn_ckde = pd.read_csv('results_pc_spbn_ckde.csv')

    df_hc_gbn = df_hc_gbn.set_index('Dataset')
    df_hc_kdebn = df_hc_kdebn.set_index('Dataset')
    df_hc_spbn = df_hc_spbn.set_index('Dataset')
    df_hc_spbn_ckde = df_hc_spbn_ckde.set_index('Dataset')

    df_pc_gbn = df_pc_gbn.set_index('Dataset')
    df_pc_kdebn = df_pc_kdebn.set_index('Dataset')
    df_pc_spbn = df_pc_spbn.set_index('Dataset')
    df_pc_spbn_ckde = df_pc_spbn_ckde.set_index('Dataset')

    df_algorithms = pd.DataFrame(index=df_num_instances.index)

    df_algorithms['N'] = df_num_instances['N']

    df_algorithms['SPBN_CKDE_Validation_10_5'] = df_hc_spbn_ckde['SPBN_CKDE_Validation_10_5']
    df_algorithms['KDEBN_5'] = df_hc_kdebn.loc[:, 'KDEBN_Validation_10_5']
    df_algorithms['BIC'] = df_hc_gbn['BIC']
    df_algorithms['BGe'] = df_hc_gbn['BGe']
    df_algorithms['SPBN_CKDE_PC_LC_10_5'] = df_pc_spbn_ckde['SPBN_CKDE_PC_LC_10_5']
    df_algorithms['SPBN_CKDE_PC_RCOT_10_5'] = df_pc_spbn_ckde['SPBN_CKDE_PC_RCOT_10_5']
    df_algorithms['KDEBN_PC_LC'] = df_pc_kdebn.loc[:, 'KDEBN_PC_LC']
    df_algorithms['KDEBN_PC_RCOT'] = df_pc_kdebn.loc[:, 'KDEBN_PC_RCOT']
    df_algorithms['GBN_PC_LC'] = df_pc_gbn['GBN_PC_LC']
    df_algorithms['GBN_PC_RCOT'] = df_pc_gbn['GBN_PC_RCOT']

    df_algorithms = df_algorithms[df_algorithms['N'] < 500]
    df_algorithms = df_algorithms.drop('N', axis=1)
    rank = df_algorithms.rank(axis=1, ascending=False)
    avgranks = rank.mean().to_numpy()
    names = rank.columns.values

    print_apv(avgranks, names)
    input("Press [Enter]:")

    names = [rename_dict[s] for s in names]

    plot_cd_diagram.graph_ranks(avgranks, names, df_algorithms.shape[0], posthoc_method="cd")
    tikzplotlib.save("plots/Nemenyi.tex", standalone=True, axis_width="14cm", axis_height="5cm")

    plot_cd_diagram.graph_ranks(avgranks, names, df_algorithms.shape[0], posthoc_method="holm")
    tikzplotlib.save("plots/Holm.tex", standalone=True, axis_width="14cm", axis_height="5cm")

    plot_cd_diagram.graph_ranks(avgranks, names, df_algorithms.shape[0], posthoc_method="bergmann")
    tikzplotlib.save("plots/Bergmann.tex", standalone=True, axis_width="14cm", axis_height="5cm")

    os.chdir("plots")
    process = subprocess.Popen('pdflatex Nemenyi.tex'.split())
    process.wait()
    process = subprocess.Popen('pdflatex Holm.tex'.split())
    process.wait()
    process = subprocess.Popen('pdflatex Bergmann.tex'.split())
    process.wait()
    process = subprocess.Popen('evince Bergmann.pdf'.split())
    process.wait()
    os.chdir("..")
    return df_algorithms

def kdeness_ckde():
    folds = experiments_helper.TRAINING_FOLDS
    patience = experiments_helper.PATIENCE

    files = experiments_helper.find_crossvalidation_datasets()
    valid_files = [f for f in files if experiments_helper.validate_dataset(f, folds) is not None]

    n_ckde = np.full((len(valid_files), len(folds), 3, 10), np.nan)
    datasets = []
    n_vars = []
    for idx_file, file in enumerate(valid_files):
        x = experiments_helper.validate_dataset(file, experiments_helper.TRAINING_FOLDS)
        dataset, result_folder = x

        basefolder = os.path.basename(os.path.dirname(file))
        datasets.append(basefolder)
        n_vars.append(dataset.shape[1])

        for idx_f, f in enumerate(experiments_helper.TRAINING_FOLDS):
            for idx_fold in range(10):
                models_folder = result_folder + '/HillClimbing/SPBN_CKDE/Validation_' + str(f) + '_5/' + str(idx_fold)
                all_models = sorted(glob.glob(models_folder + '/*.pickle'))
                final_model = load(all_models[-1])

                n_ckde[idx_file, idx_f, 0, idx_fold] = \
                    sum(map(lambda kv: kv[1] == NodeType.CKDE, final_model.node_types().items()))

        for idx_f, f in enumerate(experiments_helper.TRAINING_FOLDS):
            for idx_fold in range(10):
                models_folder = result_folder + '/PC/SPBN_CKDE/LinearCorrelation/Validation_' + str(f) + '_5/' + str(idx_fold)
                all_models = sorted(glob.glob(models_folder + '/*.pickle'))
                final_model = load(all_models[-1])

                n_ckde[idx_file, idx_f, 1, idx_fold] = \
                    sum(map(lambda kv: kv[1] == NodeType.CKDE, final_model.node_types().items()))

        for idx_f, f in enumerate(experiments_helper.TRAINING_FOLDS):
            for idx_fold in range(10):
                models_folder = result_folder + '/PC/SPBN_CKDE/RCoT/Validation_' + str(f) + '_5/' + str(idx_fold)
                all_models = sorted(glob.glob(models_folder + '/*.pickle'))
                final_model = load(all_models[-1])

                n_ckde[idx_file, idx_f, 2, idx_fold] = \
                    sum(map(lambda kv: kv[1] == NodeType.CKDE, final_model.node_types().items()))



    mean_ckde = np.mean(n_ckde, axis=3).reshape(len(valid_files), -1)
    algorithms = ["HC", "PC-PLC", "PC-RCoT"]
    names = ["CKDE_" + str(f) + "_" + algorithm for f in folds for algorithm in algorithms]

    df = pd.DataFrame(mean_ckde, columns=names, index=datasets)
    df['n_vars'] = n_vars
    for f in folds:
        for algorithm in algorithms:
            df['%CKDE_' + str(f) + "_" + algorithm] = df.loc[:,'CKDE_' + str(f) + "_" + algorithm] / df.loc[:, 'n_vars']


    N = df.shape[0]
    ind = np.arange(N)
    num_bars = len(folds) * len(algorithms)
    width = (1 - 0.3) / num_bars

    fig = plt.figure()
    ax = fig.add_subplot(111)

    offset = 0

    b = []

    color = {algorithms[0] : "#729CF5", algorithms[1]: "#FFB346", algorithms[2]: "#B5EA7F"}
    for f in folds:
        for algorithm in algorithms:
            t = ax.bar(ind+width*offset, df['%CKDE_' + str(f) + "_" + algorithm].to_numpy(), width,
                       align='edge', linewidth=0.5, edgecolor="black", color=color[algorithm])
            offset += 1
            b.append(t)

    ax.set_ylabel('Ratio of CKDE variables')
    ax.set_xticks(ind + (1 - 0.3) / 2)
    ax.set_xticklabels(df.index)
    ax.tick_params(axis='x', rotation=90)

    plt.legend([t[0] for t in b], algorithms)
    tikzplotlib.save("plots/kdeness.tex", standalone=True, axis_width="25cm", axis_height="10cm")

def datasets_table():
    files = experiments_helper.find_crossvalidation_datasets()
    valid_files = [f for f in files if experiments_helper.validate_dataset(f, [2, 3, 5, 10]) is not None]

    for idx_file, file in enumerate(valid_files):
        dataset, result_folder = experiments_helper.validate_dataset(file, [2, 3, 5, 10])
        basefolder = os.path.basename(os.path.dirname(file))

        print(basefolder + " " + str(dataset.shape[0]) + "x" + str(dataset.shape[1]))



if __name__ == '__main__':
    rename_dict = {
        'SPBN_CKDE_Validation_10_5': r'SPBN HC',
        'KDEBN_5': r'KDEBN HC',
        'BIC': r'GBN BIC',
        'BGe': r'GBN BGe',
        'SPBN_CKDE_PC_LC_10_5': r'SPBN PC-PLC',
        'SPBN_CKDE_PC_RCOT_10_5': r'SPBN PC-RCoT',
        'KDEBN_PC_LC': r'KDEBN PC-PLC',
        'KDEBN_PC_RCOT': r'KDEBN PC-RCoT',
        'GBN_PC_LC': r'GBN PC-PLC',
        'GBN_PC_RCOT': r'GBN PC-RCoT',
    }
    
    # latex = plot_cd_diagrams(rename_dict)
    kdeness_ckde()