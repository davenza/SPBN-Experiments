import numpy as np
import pandas as pd
from pybnesian.factors import NodeType
from pybnesian.factors.continuous import LinearGaussianCPD, CKDE
from pybnesian.models import SemiparametricBN
import matplotlib.pyplot as plt
import experiments_helper

np.random.seed(0)

N_MODEL = 1000

model = SemiparametricBN([('A', 'C'), ('B', 'C'), ('C', 'D'), ('D', 'E')],
                         [('A', NodeType.LinearGaussianCPD), 
                          ('B', NodeType.CKDE), 
                          ('C', NodeType.CKDE), 
                          ('D', NodeType.LinearGaussianCPD),
                          ('E', NodeType.CKDE)])


a_cpd = LinearGaussianCPD('A', [], [0.5], 1)

b_instances = experiments_helper.sample_mixture([0.5, 0.5], [-2, 2], [1, 1], N_MODEL, seed=0)
b_instances_df = pd.DataFrame({'B': b_instances})
b_cpd = CKDE('B', [])
b_cpd.fit(b_instances_df)


c_instances_a = a_cpd.sample(N_MODEL, seed=0)
c_instances_c = c_instances_a * b_instances
c_instances_c[c_instances_c <= 0] -= 2
c_instances_c[c_instances_c > 0] += 2
c_instances_df = pd.DataFrame({'A': c_instances_a, 'B': b_instances, 'C': c_instances_c})
c_cpd = CKDE('C', ['A', 'B'])
c_cpd.fit(c_instances_df)

d_cpd = LinearGaussianCPD('D', ['C'], [-2.1, -0.6], 1.5)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

e_instances_d = d_cpd.sample(N_MODEL, pd.DataFrame({'C': c_instances_c}), seed=0)
e_instances_e = sigmoid(e_instances_d.to_numpy())

e_instances_df = pd.DataFrame({'D': e_instances_d.to_numpy(), 'E': e_instances_e})

e_cpd = CKDE('E', ['D'])
e_cpd.fit(e_instances_df)

model.add_cpds([a_cpd, b_cpd, c_cpd, d_cpd, e_cpd])
model.save('true_model.pickle', include_cpd=True)

dataset200 = model.sample(200, seed=0)
dataset200.to_pandas().to_csv("synthetic_200.csv", index=False)

dataset2000 = model.sample(2000, seed=1)
dataset2000.to_pandas().to_csv("synthetic_2000.csv", index=False)

dataset10000 = model.sample(10000, seed=2)
dataset10000.to_pandas().to_csv("synthetic_10000.csv", index=False)

dataset_test = model.sample(1000, seed=3)
dataset_test.to_pandas().to_csv("synthetic_test.csv", index=False)