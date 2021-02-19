import numpy as np
np.random.seed(0)
from pybnesian import load


ecoli70_true = load('ecoli70.pickle')

ecoli70_200 = ecoli70_true.sample(200, seed=0)
ecoli70_200.to_pandas().to_csv("ecoli70_200.csv", index=False)
ecoli70_2000 = ecoli70_true.sample(2000, seed=1)
ecoli70_2000.to_pandas().to_csv("ecoli70_2000.csv", index=False)
ecoli70_10000 = ecoli70_true.sample(10000, seed=2)
ecoli70_10000.to_pandas().to_csv("ecoli70_10000.csv", index=False)
ecoli70_test = ecoli70_true.sample(1000, seed=3)
ecoli70_test.to_pandas().to_csv("ecoli70_test.csv", index=False)

magic_niab_true = load('magic_niab.pickle')

magic_niab_200 = magic_niab_true.sample(200, seed=0)
magic_niab_200.to_pandas().to_csv("magic_niab_200.csv", index=False)
magic_niab_2000 = magic_niab_true.sample(2000, seed=1)
magic_niab_2000.to_pandas().to_csv("magic_niab_2000.csv", index=False)
magic_niab_10000 = magic_niab_true.sample(10000, seed=2)
magic_niab_10000.to_pandas().to_csv("magic_niab_10000.csv", index=False)
magic_niab_test = magic_niab_true.sample(1000, seed=3)
magic_niab_test.to_pandas().to_csv("magic_niab_test.csv", index=False)

magic_irri_true = load('magic_irri.pickle')

magic_irri_200 = magic_irri_true.sample(200, seed=0)
magic_irri_200.to_pandas().to_csv("magic_irri_200.csv", index=False)
magic_irri_2000 = magic_irri_true.sample(2000, seed=1)
magic_irri_2000.to_pandas().to_csv("magic_irri_2000.csv", index=False)
magic_irri_10000 = magic_irri_true.sample(10000, seed=2)
magic_irri_10000.to_pandas().to_csv("magic_irri_10000.csv", index=False)
magic_irri_test = magic_irri_true.sample(1000, seed=3)
magic_irri_test.to_pandas().to_csv("magic_irri_test.csv", index=False)

arth150_true = load('arth150.pickle')

arth150_200 = arth150_true.sample(200, seed=0)
arth150_200.to_pandas().to_csv("arth150_200.csv", index=False)
arth150_2000 = arth150_true.sample(2000, seed=1)
arth150_2000.to_pandas().to_csv("arth150_2000.csv", index=False)
arth150_10000 = arth150_true.sample(10000, seed=2)
arth150_10000.to_pandas().to_csv("arth150_10000.csv", index=False)
arth150_test = arth150_true.sample(1000, seed=3)
arth150_test.to_pandas().to_csv("arth150_test.csv", index=False)