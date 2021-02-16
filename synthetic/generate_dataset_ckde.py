import numpy as np
import pandas as pd
from pgmpy.models import HybridContinuousModel
from pgmpy.factors.continuous import NodeType, LinearGaussianCPD, CKDE_CPD
import exp_helper

np.random.seed(0)

N_MODEL = 1000

model = HybridContinuousModel()
model.add_edges_from([('A', 'C'), ('B', 'C'), ('C', 'D'), ('D', 'E')], node_type={'A': NodeType.GAUSSIAN,
                                                                                        'B': NodeType.CKDE,
                                                                                        'C': NodeType.CKDE,
                                                                                        'D': NodeType.GAUSSIAN,
                                                                                        'E': NodeType.CKDE
                                                                                        })

a_cpd = LinearGaussianCPD('A', [0.5], 1, evidence=[])

b_instances = exp_helper.sample_mixture([0.5, 0.5], [-2, 2], [1, 1], N_MODEL)
b_instances_df = pd.DataFrame({'B': b_instances})
b_cpd = CKDE_CPD('B', gaussian_cpds=[], kde_instances=b_instances_df, evidence=[])

c_gaussiancpd_a = LinearGaussianCPD('A', [-2.0, 1, 0], 2, evidence=['C', 'B'])
c_instances = exp_helper.sample_multivariate_mixture([0.7, 0.3], [[-1, -1], [1, 1]],
                                                     [
                                                         [[1, 1],
                                                          [1, 1]],
                                                         [[2, -0.5],
                                                          [-0.5, 0.5]]
                                                     ], N_MODEL)
c_instances_df = pd.DataFrame(c_instances, columns=['C', 'B'])
c_cpd = CKDE_CPD('C', gaussian_cpds=[c_gaussiancpd_a], kde_instances=c_instances_df,
                 evidence=['A', 'B'], evidence_type={'A': NodeType.GAUSSIAN, 'B': NodeType.CKDE})

d_cpd = LinearGaussianCPD('D', [-2.1, -0.6], 1.5, evidence=['C'])

e_gaussian_cpd_d = LinearGaussianCPD('D', [0, 1.2], 0.2, evidence=['E'])
e_instances = exp_helper.sample_mixture([0.2, 0.4, 0.4], [-3, 1, 6], [1, 2, 1], N_MODEL)
e_instances_df = pd.DataFrame({'E': e_instances})
e_cpd = CKDE_CPD('E', gaussian_cpds=[e_gaussian_cpd_d], kde_instances=e_instances_df,
                 evidence=['D'], evidence_type={'D': NodeType.GAUSSIAN})

model.add_cpds(a_cpd, b_cpd, c_cpd, d_cpd, e_cpd)

model.save_model('true_model.pkl', save_parameters=True)

dataset200 = model.sample_dataset(200)
dataset200.to_csv("synthetic_200.csv", index=False)

dataset2000 = model.sample_dataset(2000)
dataset2000.to_csv("synthetic_2000.csv", index=False)

dataset10000 = model.sample_dataset(10000)
dataset10000.to_csv("synthetic_10000.csv", index=False)