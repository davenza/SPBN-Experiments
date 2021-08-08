import pandas as pd
from pybnesian import load
from pybnesian.factors import NodeType
from pybnesian.learning.scores import ValidatedLikelihood
from pybnesian.models import SemiparametricBN
import glob

df_10000 = pd.read_csv('synthetic_10000.csv')
df_test = pd.read_csv('synthetic_test.csv')

models = sorted(glob.glob('models/10000/HillClimbing/SPBN_CKDE/0/*.pickle'))

vl = ValidatedLikelihood(df_10000, k=10, seed=0)

node_types = [(name, NodeType.CKDE) for name in df_10000.columns.values]
start_model = SemiparametricBN(list(df_10000.columns.values), node_types)

print("Start model")
print("\tTraining score: " + str(vl.score(start_model)))
print("\tValidation score: " + str(vl.vscore(start_model)))

start_model.fit(df_10000)
print("\tTest score: " + str(start_model.slogl(df_test)))

for m in models:
    bn = load(m)
    print("Model " + m)
    print("\tTraining score: " + str(vl.score(bn)))
    print("\tValidation score: " + str(vl.vscore(bn)))

    bn.fit(df_10000)
    print("\tTest score: " + str(bn.slogl(df_test)))
