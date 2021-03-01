import numpy as np
from scipy.stats import norm
import pandas as pd
from pybnesian.factors import NodeType
from pybnesian.factors.continuous import LinearGaussianCPD, CKDE
from pybnesian.models import SemiparametricBN
import matplotlib.pyplot as plt
import experiments_helper

np.random.seed(0)

N_MODEL = 10000

# model = SemiparametricBN([('A', 'C'), ('B', 'C'), ('C', 'D'), ('D', 'E')],
#                          [('A', NodeType.LinearGaussianCPD), 
#                           ('B', NodeType.CKDE), 
#                           ('C', NodeType.CKDE), 
#                           ('D', NodeType.LinearGaussianCPD),
#                           ('E', NodeType.CKDE)])

# model = SemiparametricBN([('A', 'C'), ('B', 'C'), ('C', 'D'), ('D', 'E')],
#                          [('A', NodeType.LinearGaussianCPD),
#                           ('B', NodeType.CKDE),
#                           ('C', NodeType.CKDE),
#                           ('D', NodeType.LinearGaussianCPD),
#                           ('E', NodeType.CKDE)])


# a_cpd = LinearGaussianCPD('A', [], [6], 1)

# b_instances = experiments_helper.sample_mixture([0.5, 0.5], [-2, 2], [1, 1], N_MODEL, seed=1)
# b_instances_df = pd.DataFrame({'B': b_instances})
# b_cpd = CKDE('B', [])
# b_cpd.fit(b_instances_df)


# c_instances_a = a_cpd.sample(N_MODEL, seed=2)
# c_instances_b = b_cpd.sample(N_MODEL, seed=3)
# c_instances_c = c_instances_a.to_pandas()**2 * b_instances + np.random.normal(size=N_MODEL)
# # c_instances_c = c_instances_a.to_pandas()**2 + c_instances_b.to_pandas() + np.random.normal(size=N_MODEL)
# # c_instances_c[c_instances_c <= 0] -= 2
# # c_instances_c[c_instances_c > 0] += 3
# c_instances_df = pd.DataFrame({'A': c_instances_a, 'B': b_instances, 'C': c_instances_c})
# # c_instances_df.plot.density()
# # plt.show()
# c_cpd = CKDE('C', ['A', 'B'])
# c_cpd.fit(c_instances_df)

# d_cpd = LinearGaussianCPD('D', ['C'], [10, 0.8], 0.5)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# e_instances_d = d_cpd.sample(N_MODEL, pd.DataFrame({'C': c_instances_c}), seed=0)
# e_instances_e = sigmoid(e_instances_d.to_numpy()) + np.random.normal(0, 0.5)

# e_instances_df = pd.DataFrame({'D': e_instances_d.to_numpy(), 'E': e_instances_e})

# e_cpd = CKDE('E', ['D'])
# e_cpd.fit(e_instances_df)

# # model.add_cpds([a_cpd, b_cpd, c_cpd, d_cpd, e_cpd])
# model.add_cpds([a_cpd, b_cpd, c_cpd, d_cpd, e_cpd]) 
# # model.save('true_model.pickle', include_cpd=True)

# dataset200 = model.sample(200, seed=0)
# # dataset200.to_pandas().to_csv("synthetic_200.csv", index=False)

# dataset2000 = model.sample(2000, seed=1)
# # dataset2000.to_pandas().to_csv("synthetic_2000.csv", index=False)

# dataset10000 = model.sample(10000, seed=2)
# # dataset10000.to_pandas().to_csv("synthetic_10000.csv", index=False)

# dataset_test = model.sample(1000, seed=3)
# # dataset_test.to_pandas().to_csv("synthetic_test.csv", index=False)

from pybnesian.learning.independences import LinearCorrelation, RCoT

def generate_dataset(size, seed = 0):
  np.random.seed(seed)
  def generate_A():
    return pd.Series(np.random.normal(loc=6, scale=1, size=size), name="A")

  def generate_B():
    return pd.Series(experiments_helper.sample_mixture([0.5, 0.5], [-2, 2], [1, 1], size), name="A")

  def generate_C(evidence):
    return pd.Series(evidence['A']**2 * evidence['B'] + np.random.normal(size=size), name="C")
  
  def generate_D(evidence):
    return pd.Series(10 + 0.8*evidence['C'] + np.random.normal(loc=0, scale=np.sqrt(0.5), size=size), name="D")

  def generate_E(evidence):
    return pd.Series(sigmoid(evidence['D']) + np.random.normal(loc=0, scale=np.sqrt(0.5), size=size), name="E")

  df = pd.DataFrame()
  df['A'] = generate_A()
  df['B'] = generate_B()
  df['C'] = generate_C(df)
  df['D'] = generate_D(df)
  df['E'] = generate_E(df)

  return df

def slogl_model(df):
  slogl = norm(loc=6, scale=1).logpdf(df['A']).sum()
  slogl += np.log(0.5 * norm(loc=-2, scale=1).pdf(df['B']) + 0.5 * norm(loc=2, scale=1).pdf(df['B'])).sum()
  slogl += norm(loc=df['A']**2 * df['B'], scale=1).logpdf(df['C']).sum()
  slogl += norm(loc=10 + 0.8*df['C'], scale=np.sqrt(0.5)).logpdf(df['D']).sum()
  slogl += norm(loc=sigmoid(df['D']), scale=np.sqrt(0.5)).logpdf(df['E']).sum()

  return slogl


dataset200 = generate_dataset(200, seed=0)
dataset200.to_csv("synthetic_200.csv", index=False)

dataset2000 = generate_dataset(2000, seed=1)
dataset2000.to_csv("synthetic_2000.csv", index=False)

dataset10000 = generate_dataset(10000, seed=2)
dataset10000.to_csv("synthetic_10000.csv", index=False)

dataset_test = generate_dataset(1000, seed=3)
dataset_test.to_csv("synthetic_test.csv", index=False)

def print_interesting_independences(test):
  pvalue = test.pvalue("A", "B")
  print("A _|_ B " + str(pvalue) + " " + ("OK" if pvalue > 0.05 else "FAIL"))
  pvalue = test.pvalue("A", "B", "C")
  print("A _|_ B | C " + str(pvalue) + " " + ("OK" if pvalue < 0.05 else "FAIL"))
  pvalue = test.pvalue("C", "D")
  print("C _|_ D " + str(pvalue) + " " + ("OK" if pvalue < 0.05 else "FAIL"))
  pvalue = test.pvalue("A", "D")
  print("A _|_ D " + str(pvalue) + " " + ("OK" if pvalue < 0.05 else "FAIL"))
  pvalue = test.pvalue("A", "D", "C")
  print("A _|_ D | C " + str(pvalue) + " " + ("OK" if pvalue > 0.05 else "FAIL"))
  pvalue = test.pvalue("B", "D")
  print("B _|_ D " + str(pvalue) + " " + ("OK" if pvalue < 0.05 else "FAIL"))
  pvalue = test.pvalue("B", "D", "C")
  print("B _|_ D | C " + str(pvalue) + " " + ("OK" if pvalue > 0.05 else "FAIL"))
  pvalue = test.pvalue("C", "E")
  print("C _|_ E " + str(pvalue) + " " + ("OK" if pvalue < 0.05 else "FAIL"))
  pvalue = test.pvalue("C", "E", "D")
  print("C _|_ E | D " + str(pvalue) + " " + ("OK" if pvalue > 0.05 else "FAIL"))


lc = LinearCorrelation(dataset10000)
rc = RCoT(dataset10000)
print("LinearCorrelation:")
print("-------------------------")
print_interesting_independences(lc)
print("RCoT:")
print("-------------------------")
print_interesting_independences(rc)