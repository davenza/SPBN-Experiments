import numpy as np
from scipy.stats import norm
import pandas as pd
from pybnesian.factors import NodeType
from pybnesian.factors.continuous import LinearGaussianCPD, CKDE
from pybnesian.models import SemiparametricBN
import matplotlib.pyplot as plt
import experiments_helper

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def generate_dataset(size, seed = 0):
  np.random.seed(seed)
  def generate_A():
    return pd.Series(np.random.normal(loc=6, scale=1, size=size), name="A")

  def generate_B():
    return pd.Series(experiments_helper.sample_mixture([0.5, 0.5], [-2, 2], [1, 1], size), name="A")

  def generate_C(evidence):
    return pd.Series(evidence['A'] * evidence['B'] + np.random.normal(size=size), name="C")
  
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
  slogl += norm(loc=df['A'] * df['B'], scale=1).logpdf(df['C']).sum()
  slogl += norm(loc=10 + 0.8*df['C'], scale=np.sqrt(0.5)).logpdf(df['D']).sum()
  slogl += norm(loc=sigmoid(df['D']), scale=np.sqrt(0.5)).logpdf(df['E']).sum()

  return slogl


for i in range(experiments_helper.NUM_SIMULATIONS):
  dataset200 = generate_dataset(200, seed=i*100)
  dataset200.to_csv("data/synthetic_" + str(i).zfill(3) + "_200.csv", index=False)

  dataset2000 = generate_dataset(2000, seed=1 + (i*100))
  dataset2000.to_csv("data/synthetic_" + str(i).zfill(3) + "_2000.csv", index=False)

  dataset10000 = generate_dataset(10000, seed=2 + (i*100))
  dataset10000.to_csv("data/synthetic_" + str(i).zfill(3) + "_10000.csv", index=False)

  dataset_test = generate_dataset(1000, seed=3 + (i*100))
  dataset_test.to_csv("data/synthetic_" + str(i).zfill(3) + "_test.csv", index=False)


model = SemiparametricBN([('A', 'C'), ('B', 'C'), ('C', 'D'), ('D', 'E')],
                         [('A', NodeType.LinearGaussianCPD), 
                          ('B', NodeType.CKDE), 
                          ('C', NodeType.CKDE), 
                          ('D', NodeType.LinearGaussianCPD),
                          ('E', NodeType.CKDE)])

model.save("true_model.pickle")