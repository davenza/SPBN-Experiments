import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_cd_diagram

import subprocess
import glob

import os
import experiments_helper

from pgmpy.models import HybridContinuousModel
from pgmpy.factors.continuous import NodeType

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
    df_bn = pd.read_csv('cv_results_bn.csv')
    df_gmm = pd.read_csv('cv_results_gmm.csv')
    df_kde = pd.read_csv('cv_results_kde.csv')
    df_kdebn = pd.read_csv('cv_results_kdenetwork.csv')
    df_spbn_strict = pd.read_csv('cv_results_spbn_strict.csv')

    # df_bn = df_bn.loc[:, ['Dataset', 'CKDE_Validation_10_0', 'CKDE_Validation_10_5', 'Gaussian_Validation_10_0',
    #                       'Gaussian_Validation_10_5', 'BIC', 'BGe']]
    # df_bn = df_bn.loc[:, ['Dataset', 'Gaussian_Validation_10_0', 'Gaussian_Validation_10_5', 'BIC', 'BGe']]
    df_bn = df_bn.loc[:, ['Dataset', 'CKDE_Validation_10_0', 'CKDE_Validation_10_5', 'BIC', 'BGe',
                          'Gaussian_Validation_10_0', 'Gaussian_Validation_10_5']]


    df_bn = df_bn.set_index('Dataset')
    df_gmm = df_gmm.set_index('Dataset')
    df_kde = df_kde.set_index('Dataset')
    df_kdebn = df_kdebn.set_index('Dataset')
    df_spbn_strict = df_spbn_strict.set_index('Dataset')

    # df_concat = pd.concat([df_bn, df_gmm], axis=1)
    df_algorithms = pd.DataFrame(index=df_bn.index)

    # df_algorithms['CKDE_Validation_10_0'] = df_bn['CKDE_Validation_10_0']
    # df_algorithms['CKDE_Validation_10_5'] = df_bn['CKDE_Validation_10_5']
    df_algorithms['BIC'] = df_bn['BIC']
    df_algorithms['BGe'] = df_bn['BGe']
    df_algorithms['Gaussian_Validation_10_0'] = df_bn['Gaussian_Validation_10_0']
    df_algorithms['Gaussian_Validation_10_5'] = df_bn['Gaussian_Validation_10_5']
    # df_algorithms['GMM'] = df_gmm.max(1)
    # df_algorithms['KDE'] = df_kde.loc[:, 'KDE']
    df_algorithms['KDEBN_0'] = df_kdebn.loc[:, 'KDEBN_Validation_10_0']
    df_algorithms['KDEBN_5'] = df_kdebn.loc[:, 'KDEBN_Validation_10_5']
    df_algorithms['SPBN_STRICT_10_0'] = df_spbn_strict.loc[:, 'SPBN_STRICT_10_0']
    df_algorithms['SPBN_STRICT_10_5'] = df_spbn_strict.loc[:, 'SPBN_STRICT_10_5']

    df_algorithms = df_algorithms.drop("Haberman", errors='ignore')
    df_algorithms = df_algorithms.drop("Thyroid", errors='ignore')
    df_algorithms = df_algorithms.drop("Transfusion", errors='ignore')

    # print(df_algorithms[['SPBN_STRICT_10_5', 'SPBN_STRICT_10_0', 'KDEBN_5', 'KDEBN_0']])
    # print(df_algorithms[['SPBN_STRICT_10_5', 'SPBN_STRICT_10_0', 'KDEBN_5', 'KDEBN_0']].rank(axis=1, ascending=False))

    rank = df_algorithms.rank(axis=1, ascending=False)
    avgranks = rank.mean().to_numpy()
    names = rank.columns.values

    print_apv(avgranks, names)
    input("Press [Enter]:")

    names = [rename_dict[s] for s in names]

    plot_cd_diagram.graph_ranks(avgranks, names, df_algorithms.shape[0], posthoc_method="cd")
    tikzplotlib.save("plots/Nemenyi_spbn.tex", standalone=True, axis_width="12cm", axis_height="5cm")

    plot_cd_diagram.graph_ranks(avgranks, names, df_algorithms.shape[0], posthoc_method="holm")
    tikzplotlib.save("plots/Holm_spbn.tex", standalone=True, axis_width="12cm", axis_height="5cm")

    plot_cd_diagram.graph_ranks(avgranks, names, df_algorithms.shape[0], posthoc_method="bergmann")
    tikzplotlib.save("plots/Bergmann_spbn.tex", standalone=True, axis_width="12cm", axis_height="5cm")

    os.chdir("plots")
    process = subprocess.Popen('pdflatex Nemenyi_spbn.tex'.split())
    process.wait()
    process = subprocess.Popen('pdflatex Holm_spbn.tex'.split())
    process.wait()
    process = subprocess.Popen('pdflatex Bergmann_spbn.tex'.split())
    process.wait()
    process = subprocess.Popen('evince Bergmann_spbn.pdf'.split())
    process.wait()
    os.chdir("..")
    return df_algorithms

def kdeness_ckde():
    folds = [10]
    patience = [0, 5]

    files = experiments_helper.find_crossvalidation_datasets()
    valid_files = [f for f in files if experiments_helper.validate_dataset(f, [2, 3, 5, 10]) is not None]

    n_ckde = np.full((len(valid_files), len(folds), len(patience), 10), np.nan)
    datasets = []
    n_vars = []
    for idx_file, file in enumerate(valid_files):
        x = experiments_helper.validate_dataset(file, [2, 3, 5, 10])
        dataset, result_folder = x

        basefolder = os.path.basename(os.path.dirname(file))
        datasets.append(basefolder)
        n_vars.append(dataset.shape[1])

        for idx_f, f in enumerate(folds):
            for idx_p, p in enumerate(patience):
                for idx_fold in range(10):
                    models_folder = result_folder + '/SPBN_Strict/Validation_' + str(f) + "_" + str(p) + '/' + str(idx_fold)
                    all_models = sorted(glob.glob(models_folder + '/*.pkl'))
                    final_model = HybridContinuousModel.load_model(all_models[-1])

                    n_ckde[idx_file, idx_f, idx_p, idx_fold] = \
                        sum(map(lambda kv: kv[1] == NodeType.SPBN_STRICT, final_model.node_type.items()))
                        # sum(map(lambda kv: kv[1] == NodeType.CKDE, final_model.node_type.items()))

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
    tikzplotlib.save("plots/kdeness_spbn.tex", standalone=True, axis_width="25cm", axis_height="10cm")

def datasets_table():
    files = experiments_helper.find_crossvalidation_datasets()
    valid_files = [f for f in files if experiments_helper.validate_dataset(f, [2, 3, 5, 10]) is not None]

    for idx_file, file in enumerate(valid_files):
        dataset, result_folder = experiments_helper.validate_dataset(file, [2, 3, 5, 10])
        basefolder = os.path.basename(os.path.dirname(file))

        print(basefolder + " " + str(dataset.shape[0]) + "x" + str(dataset.shape[1]))



if __name__ == '__main__':
    rename_dict = {
        'CKDE_Validation_10_0': r'CKDE $\lambda = 0$',
        'CKDE_Validation_10_5': r'CKDE $\lambda = 5$',
        'Gaussian_Validation_10_0': r'GBN $\lambda = 0$',
        'Gaussian_Validation_10_5': r'GBN $\lambda = 5$',
        'BIC': r'GBN BIC',
        'BGe': r'GBN BGe',
        'KDEBN_0': r'KDEBN $\lambda = 0$',
        'KDEBN_5': r'KDEBN $\lambda = 5$',
        'SPBN_STRICT_10_0': r'SPBN $\lambda = 0$',
        'SPBN_STRICT_10_5': r'SPBN $\lambda = 5$',
    }
    # latex = plot_cd_diagrams(rename_dict)

    kdeness_ckde()