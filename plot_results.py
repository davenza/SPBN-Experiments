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

def print_apv(avgranks, names, type="bergmann"):
    if type == "bergmann":
        apv = adjusted_pvalues.bergmann_hommel(avgranks, names)
    elif type == "holm":
        apv = adjusted_pvalues.holm(avgranks, names)

    for (pvalue, (alg1, alg2)) in apv:
        print(alg1 + " vs " + alg2  + ": " + str(pvalue))

def plot_cd_diagrams(rename_dict):
    df_hc_gbn = pd.read_csv('results_hc_gbn.csv')
    df_hc_kdebn = pd.read_csv('results_hc_kdebn.csv')
    df_hc_spbn = pd.read_csv('results_hc_spbn.csv')

    df_pc_gbn = pd.read_csv('results_pc_gbn.csv')
    df_pc_kdebn = pd.read_csv('results_pc_kdebn.csv')
    df_pc_spbn = pd.read_csv('results_pc_spbn.csv')

    df_hc_gbn = df_hc_gbn.set_index('Dataset')
    df_hc_kdebn = df_hc_kdebn.set_index('Dataset')
    df_hc_spbn = df_hc_spbn.set_index('Dataset')

    df_pc_gbn = df_pc_gbn.set_index('Dataset')
    df_pc_kdebn = df_pc_kdebn.set_index('Dataset')
    df_pc_spbn = df_pc_spbn.set_index('Dataset')

    df_algorithms = pd.DataFrame(index=df_hc_gbn.index)

    df_algorithms['SPBN_Validation_10_0'] = df_hc_spbn['SPBN_Validation_10_0']
    df_algorithms['SPBN_Validation_10_5'] = df_hc_spbn['SPBN_Validation_10_5']
    df_algorithms['KDEBN_0'] = df_hc_kdebn.loc[:, 'KDEBN_Validation_10_0']
    df_algorithms['KDEBN_5'] = df_hc_kdebn.loc[:, 'KDEBN_Validation_10_5']
    df_algorithms['GBN_Validation_10_0'] = df_hc_gbn['GBN_Validation_10_0']
    df_algorithms['GBN_Validation_10_5'] = df_hc_gbn['GBN_Validation_10_5']
    df_algorithms['BIC'] = df_hc_gbn['BIC']
    df_algorithms['BGe'] = df_hc_gbn['BGe']

    df_algorithms['SPBN_PC_LC_10_0'] = df_pc_spbn['SPBN_PC_LC_10_0']
    df_algorithms['SPBN_PC_LC_10_5'] = df_pc_spbn['SPBN_PC_LC_10_5']
    df_algorithms['SPBN_PC_RCOT_10_0'] = df_pc_spbn['SPBN_PC_RCOT_10_0']
    df_algorithms['SPBN_PC_RCOT_10_5'] = df_pc_spbn['SPBN_PC_RCOT_10_5']
    df_algorithms['KDEBN_PC_LC'] = df_pc_kdebn.loc[:, 'KDEBN_PC_LC']
    df_algorithms['KDEBN_PC_RCOT'] = df_pc_kdebn.loc[:, 'KDEBN_PC_RCOT']
    df_algorithms['GBN_PC_LC'] = df_pc_gbn['GBN_PC_LC']
    df_algorithms['GBN_PC_RCOT'] = df_pc_gbn['GBN_PC_RCOT']

    rank = df_algorithms.rank(axis=1, ascending=False)
    avgranks = rank.mean().to_numpy()
    names = rank.columns.values

    print_apv(avgranks, names)
    input()

    names = [rename_dict[s] for s in names]

    plot_cd_diagram.graph_ranks(avgranks, names, df_algorithms.shape[0], posthoc_method="cd")
    tikzplotlib.save("plots/Nemenyi.tex", standalone=True, axis_width="12cm", axis_height="5cm")

    plot_cd_diagram.graph_ranks(avgranks, names, df_algorithms.shape[0], posthoc_method="holm")
    tikzplotlib.save("plots/Holm.tex", standalone=True, axis_width="12cm", axis_height="5cm")

    plot_cd_diagram.graph_ranks(avgranks, names, df_algorithms.shape[0], posthoc_method="bergmann")
    tikzplotlib.save("plots/Bergmann.tex", standalone=True, axis_width="12cm", axis_height="5cm")

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
    valid_files = [f for f in files if experiments_helper.validate_dataset(f, experiments_helper.TRAINING_FOLDS) is not None]

    n_ckde = np.full((len(valid_files), len(folds), len(patience), experiments_helper.EVALUATION_FOLDS), np.nan)
    datasets = []
    n_vars = []
    for idx_file, file in enumerate(valid_files):
        x = experiments_helper.validate_dataset(file, experiments_helper.TRAINING_FOLDS)
        dataset, result_folder = x

        basefolder = os.path.basename(os.path.dirname(file))
        datasets.append(basefolder)
        n_vars.append(dataset.shape[1])

        for idx_f, f in enumerate(folds):
            for idx_p, p in enumerate(patience):
                for idx_fold in range(experiments_helper.EVALUATION_FOLDS):
                    models_folder = result_folder + 'HillClimbing/SPBN/Validation_' + str(f) + "_" + str(p) + '/' + str(idx_fold)
                    all_models = sorted(glob.glob(models_folder + '/*.pkl'))
                    final_model = load_model(all_models[-1])

                    n_ckde[idx_file, idx_f, idx_p, idx_fold] = \
                        sum(map(lambda kv: kv == NodeType.CKDE, [final_model.node_type(node) for node in final_model.nodes()]))

    mean_ckde = np.mean(n_ckde, axis=3).reshape(len(valid_files), -1)
    names = ["CKDE_" + str(f) + "_" + str(p) for f in folds for p in patience]

    df = pd.DataFrame(mean_ckde, columns=names, index=datasets)
    df['n_vars'] = n_vars
    for f in folds:
        for p in patience:
            df['%CKDE_' + str(f) + "_" + str(p)] = df.loc[:,'CKDE_' + str(f) + "_" + str(p)] / df.loc[:, 'n_vars']


    N = df.shape[0]
    ind = np.arange(N)
    num_bars = len(folds) * len(patience)
    width = (1 - 0.3) / num_bars

    fig = plt.figure()
    ax = fig.add_subplot(111)

    offset = 0

    b = []
    color = {0: "#729CF5", 5: "#FFB346"}
    for f in folds:
        for p in patience:
            t = ax.bar(ind+width*offset, df['%CKDE_' + str(f) + "_" + str(p)].to_numpy(), width,
                       align='edge', color=color[p])
            offset += 1
            b.append(t)

    ax.set_ylabel('Ratio of CKDE variables')
    ax.set_xticks(ind + (1 - 0.3) / 2)
    ax.set_xticklabels(df.index)
    ax.tick_params(axis='x', rotation=90)

    # ax.legend(tuple([t[0] for t in b]), (r'$\lambda = 0$', r'$\lambda = 5$'))
    plt.legend([t[0] for t in b], [r'$\lambda = 0$', r'$\lambda = 5$'])
    tikzplotlib.save("plots/kdeness.tex", standalone=True, axis_width="25cm", axis_height="10cm")

def datasets_table():
    files = experiments_helper.find_crossvalidation_datasets()
    valid_files = [f for f in files if experiments_helper.validate_dataset(f, experiments_helper.TRAINING_FOLDS) is not None]

    for idx_file, file in enumerate(valid_files):
        dataset, result_folder = experiments_helper.validate_dataset(file, experiments_helper.TRAINING_FOLDS)
        basefolder = os.path.basename(os.path.dirname(file))

        print(basefolder + " " + str(dataset.shape[0]) + "x" + str(dataset.shape[1]))


if __name__ == '__main__':
    rename_dict = {
        'SPBN_Validation_10_0': r'CKDE $\lambda = 0$',
        'SPBN_Validation_10_5': r'CKDE $\lambda = 5$',
        'KDEBN_0': r'KDEBN $\lambda = 0$',
        'KDEBN_5': r'KDEBN $\lambda = 5$',
        'GBN_Validation_10_0': r'GBN $\lambda = 0$',
        'GBN_Validation_10_5': r'GBN $\lambda = 5$',
        'BIC': r'GBN BIC',
        'BGe': r'GBN BGe',
        'SPBN_PC_LC_10_0': r'SPBN PC LC $\lambda = 0$',
        'SPBN_PC_LC_10_5': r'SPBN PC LC $\lambda = 5$',
        'SPBN_PC_RCOT_10_0': r'SPBN PC RCOT $\lambda = 0$',
        'SPBN_PC_RCOT_10_5': r'SPBN PC RCOT $\lambda = 5$',
        'KDEBN_PC_LC': r'KDEBN PC LC',
        'KDEBN_PC_RCOT': r'KDEBN PC RCOT',
        'GBN_PC_LC': r'GBN_PC_LC',
        'GBN_PC_RCOT': r'GBN_PC_RCOT',
    }


    
    latex = plot_cd_diagrams(rename_dict)

    # df = kdeness_ckde()
    # print(df)

    # datasets_table()