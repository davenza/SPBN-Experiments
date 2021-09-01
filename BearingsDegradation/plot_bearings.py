import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pybnesian.learning.algorithms import GreedyHillClimbing
from pybnesian.learning.operators import OperatorPool, ArcOperatorSet, ChangeNodeTypeSet
from pybnesian.learning.scores import ValidatedLikelihood
from pybnesian.models import SemiparametricBN
import hmmlearn.hmm

import tikzplotlib

def get_labels_ordered(ghmm, df):
    labels = ghmm.predict(df)

    labels0 = labels == 0
    labels1 = labels == 1
    labels2 = labels == 2

    mean0 = df.iloc[labels0,:].mean().mean()
    mean1 = df.iloc[labels1,:].mean().mean()
    mean2 = df.iloc[labels2,:].mean().mean()

    means = np.asarray([mean0, mean1, mean2])
    arg_sort = np.argsort(means)

    labels0 = labels == arg_sort[0]
    labels1 = labels == arg_sort[1]
    labels2 = labels == arg_sort[2]

    ind0 = np.where(labels0)[0]
    ind1 = np.where(labels1)[0]
    ind2 = np.where(labels2)[0]

    return labels0, labels1, labels2

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


if __name__ == "__main__":

    atrium11 = pd.read_csv("atrium11.csv")
    atrium11 = atrium11.filter(regex="BPFO|BPFI|FTF|BSF|W")
    
    atrium12 = pd.read_csv("atrium12.csv")
    atrium12 = atrium12.filter(regex="BPFO|BPFI|FTF|BSF|W")
    
    atrium13 = pd.read_csv("atrium13.csv")
    atrium13 = atrium13.filter(regex="BPFO|BPFI|FTF|BSF|W")
    
    atrium17 = pd.read_csv("atrium17.csv")
    atrium17 = atrium17.filter(regex="BPFO|BPFI|FTF|BSF|W")

    ghmm = hmmlearn.hmm.GaussianHMM(3)
    ghmm.fit(atrium11)

    labels0_11, labels1_11, labels2_11 = get_labels_ordered(ghmm, atrium11)
    labels0_12, labels1_12, labels2_12 = get_labels_ordered(ghmm, atrium12)

    plt.scatter(np.where(labels0_11)[0]*10, np.full((labels0_11.sum(),), 0))
    plt.scatter(np.where(labels1_11)[0]*10, np.full((labels1_11.sum(),), 1))
    plt.scatter(np.where(labels2_11)[0]*10, np.full((labels2_11.sum(),), 2))

    plt.gca().yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 1, 2]))
    y_values = ["Good", "Average", "Bad"]
    plt.gca().set_yticklabels(y_values)
    plt.xlabel("Duration (s)")
    plt.ylabel("Condition State")
    tikzplotlib.save("plots/Bearing_Segmentation.tex", standalone=True, axis_width="20.5cm", axis_height="5cm")

    plt.figure()

    good = pd.concat([atrium11.iloc[labels0_11,:], atrium12.iloc[labels0_12,:]])
    mid = pd.concat([atrium11.iloc[labels1_11,:], atrium12.iloc[labels1_12,:]])
    poor = pd.concat([atrium11.iloc[labels2_11,:], atrium12.iloc[labels2_12,:]])

    hc = GreedyHillClimbing()
    pool = OperatorPool([ArcOperatorSet(), ChangeNodeTypeSet()])
    start_model = SemiparametricBN(list(good.columns.values))
    vl = ValidatedLikelihood(good, k=10)
    good_model = hc.estimate(pool, vl, start_model)
    good_model.fit(good)
    
    hc = GreedyHillClimbing()
    pool = OperatorPool([ArcOperatorSet(), ChangeNodeTypeSet()])
    start_model = SemiparametricBN(list(mid.columns.values))
    vl = ValidatedLikelihood(mid, k=10)
    mid_model = hc.estimate(pool, vl, start_model)
    mid_model.fit(mid)
    
    hc = GreedyHillClimbing()
    pool = OperatorPool([ArcOperatorSet(), ChangeNodeTypeSet()])
    start_model = SemiparametricBN(list(poor.columns.values))
    vl = ValidatedLikelihood(poor, k=10)
    poor_model = hc.estimate(pool, vl, start_model)
    poor_model.fit(poor)

    logl_good = good_model.logl(atrium13)
    logl_mid = mid_model.logl(atrium13)
    logl_poor = poor_model.logl(atrium13)

    lls = np.vstack((logl_good, logl_mid, logl_poor)).T

    max_index = np.argmax(lls, axis=1)

    plt.plot(np.arange(max_index.shape[0]-29)*10, moving_average(max_index, 30))
    plt.gca().yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 1, 2]))
    y_values = ["Good", "Average", "Bad"]
    plt.gca().set_yticklabels(y_values)
    plt.xlabel("Duration (s)")
    plt.ylabel("Condition State")
    tikzplotlib.save("plots/Bearing_process.tex", standalone=True, axis_width="20cm", axis_height="10cm")

    plt.figure()

    logl_good = good_model.logl(atrium17)

    bpfo = np.zeros_like(logl_good)
    bpfi = np.zeros_like(logl_good)
    ftf = np.zeros_like(logl_good)
    bsf = np.zeros_like(logl_good)
    for n in good_model.nodes():
        cpd = good_model.cpd(n)

        if "BPFO" in n:
            bpfo += cpd.logl(atrium17)
        elif "BPFI" in n:
            bpfi += cpd.logl(atrium17)
        elif "FTF" in n:
            ftf += cpd.logl(atrium17)
        elif "BSF" in n:
            bsf += cpd.logl(atrium17)

    plt.plot(np.arange(logl_good.shape[0]-29)*10, moving_average(logl_good, 30))
    plt.plot(np.arange(bpfo.shape[0]-29)*10, moving_average(bpfo, 30))
    plt.plot(np.arange(bpfi.shape[0]-29)*10, moving_average(bpfi, 30))
    plt.plot(np.arange(ftf.shape[0]-29)*10, moving_average(ftf, 30))
    plt.plot(np.arange(bsf.shape[0]-29)*10, moving_average(bsf, 30))
    plt.xlim(21650, 22290)
    plt.ylim(-23200, 500)
    plt.legend(["Global log-likelihood", "BPFO log-likelihood", "BPFI log-likelihood", 
                "FTF log-likelihood", "BSF log-likelihood"])
    plt.xlabel("Duration (s)")
    plt.ylabel("Log-likelihood")
    tikzplotlib.save("plots/Bearing_local_defect.tex", standalone=True, axis_width="20cm", axis_height="10cm")