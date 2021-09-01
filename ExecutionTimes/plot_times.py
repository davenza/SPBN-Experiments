import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import tikzplotlib

COLOR1 = "#729CF5"
COLOR2 = "#fc9b0f"
COLOR3 = "#B5EA7F"
COLOR4 = "#ed0000"
COLOR5 = "#00524f"
COLOR6 = "#4b0066"
COLOR7 = "#ff00d0"
COLOR8 = "#000000"

INSTANCES = [200, 500, 1000, 2000, 4000, 10000]
MODELS = ["small", "medium", "large"]
COLORS = [COLOR1, COLOR2, COLOR3, COLOR4, COLOR5, COLOR6, COLOR7, COLOR8]

def extract_algorithm(name):
    df = pd.read_csv(name + "_" + MODELS[0] + ".csv")

    regex = re.compile("(.*)_10000")
    cols = list(filter(regex.match, df.columns))

    # Detect algorithms
    algorithms = []
    for c in cols:
        algorithms.append(regex.search(c).group(1))

    times = []
    for alg in algorithms:
        local_time = np.empty((len(INSTANCES)*3))

        for i, inst in enumerate(INSTANCES):
            local_time[i] = df[alg + "_" + str(inst)].mean()

        for j, m in enumerate(MODELS[1:]):
            df = pd.read_csv(name + "_" + m + ".csv")

            for i, inst in enumerate(INSTANCES):
                local_time[i + (j+1)*len(INSTANCES)] = df[alg + "_" + str(inst)].mean()

        times.append(local_time)

    return algorithms, times

def extract_info():
    spbn_names, spbn_times = extract_algorithm("HC_SPBN")
    spbn_ckde_names, spbn_ckde_times = extract_algorithm("HC_SPBN_CKDE")
    kde_names, kde_times = extract_algorithm("HC_KDEBN")
    bic_names, bic_times = extract_algorithm("HC_GBN_BIC")
    bge_names, bge_times = extract_algorithm("HC_GBN_BGe")
    pc_names, pc_times = extract_algorithm("PC")

    node_times = np.zeros_like(pc_times[0])
    number_node_times = 0

    for name, time in zip(pc_names, pc_times):
        if "NodeType" in name:
            node_times += time
            number_node_times += 1

    node_times /= number_node_times

    pc_names, pc_times = zip(*((name, time) for name, time in zip(pc_names, pc_times) if "NodeType" not in name))

    pc_names = list(pc_names)
    pc_times = list(pc_times)
    pc_names += ["PC-NodeType"]
    pc_times += [node_times]
        
    return (spbn_names + spbn_ckde_names + kde_names + bic_names + bge_names + pc_names, 
            spbn_times + spbn_ckde_times + kde_times + bic_times + bge_times + pc_times)

def plot_execution_time(rename_dict):

    algs, times = extract_info()

    N = len(INSTANCES) * len(MODELS)
    ind = np.arange(N, dtype=np.float64).reshape(len(MODELS), -1)

    # # Between dataset groups separation
    n_instances = len(INSTANCES)
    for i in range(1, len(MODELS)):
        ind[i:,:] += 0.5

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(MODELS)):
        for name, time, color in zip(algs, times, COLORS):
            reshape_time = time.reshape(len(MODELS), -1)
            if "Graph" in name:
                ax.plot(ind[i,:], reshape_time[i,:], c=color, linestyle="--")
            else:
                ax.plot(ind[i,:], reshape_time[i,:], c=color)

    ax.set_yscale('log')

    ax.set_ylabel('Execution Time (s)')
    ax.set_xticks(ind.reshape(-1))
    labels = INSTANCES * len(MODELS)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x')

    ax.xaxis.grid(which="major", color='k', linestyle='--', dashes=(5, 5), linewidth=0.7)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.arange(2.5, 16, len(INSTANCES)+0.5))
    ax2.set_xticklabels(MODELS)
    ax2.xaxis.set_ticks_position('bottom')

    for t in ax2.get_xticklabels():
        t.set_y(-0.04)

    ax2.tick_params(axis='both', which='both', length=0)

    labels = [rename_dict[alg] for alg in algs]
    ax.legend(labels)
    # plt.show()
    tikzplotlib.save("plots/plot_times.tex", standalone=True, axis_width="25cm", axis_height="12cm")


if __name__ == "__main__":

    rename_dict = {
        'SPBN': 'HC SPBN-LG',
        'SPBN_CKDE': 'HC SPBN-CKDE',
        'KDEBN': 'HC KDEBN',
        'GBN_BIC': 'GBN BIC',
        'GBN_BGe': 'GBN BGe',
        'Graph-LC': 'PC-PLC Graph',
        'Graph-RCoT': 'PC-RCoT Graph',
        'PC-NodeType': 'PC-HC-NodeType',
    }

    plot_execution_time(rename_dict)