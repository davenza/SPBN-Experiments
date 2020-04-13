import numpy as np
np.random.seed(0)
import pandas as pd
import pathlib
import glob
from pgmpy.models import HybridContinuousModel


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
        if estimated.node_type[n] != true.node_type[n]:
            hamming_value += 1

    return hamming_value

true_model = HybridContinuousModel.load_model('true_model.pkl')

test_df = true_model.sample_dataset(1000)

print("True model logl: " + str(true_model.logpdf_dataset(test_df).sum()))

df_200 = pd.read_csv('synthetic_200.csv')
df_2000 = pd.read_csv('synthetic_2000.csv')
df_10000 = pd.read_csv('synthetic_10000.csv')

patience = [0, 5]

for df, model_folder in [(df_200, 'models/200'), (df_2000, 'models/2000'), (df_10000, 'models/10000')]:
    print("Folder " + model_folder)
    for p in patience:
        result_folder = model_folder + '/' + str(p)
        pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

        all_models = sorted(glob.glob(result_folder + '/*.pkl'))
        final_model = all_models[-1]

        hcm = HybridContinuousModel.load_model(final_model)
        hcm.fit(df)

        logl = hcm.logpdf_dataset(test_df)

        print("Loglik, p " + str(p) + ": " + str(logl.sum()))
        print("SHD, p " + str(p) + ": " + str(shd(hcm, true_model)))
        print("Hamming, p " + str(p) + ": " + str(hamming(hcm, true_model)))
        print("Hamming type, p " + str(p) + ": " + str(hamming_type(hcm, true_model)))

        print()



