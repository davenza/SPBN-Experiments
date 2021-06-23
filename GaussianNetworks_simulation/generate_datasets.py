import numpy as np
np.random.seed(0)
from pybnesian import load
import experiments_helper


ecoli70_true = load('ecoli70.pickle')
magic_niab_true = load('magic_niab.pickle')
magic_irri_true = load('magic_irri.pickle')
arth150_true = load('arth150.pickle')

for i in range(experiments_helper.NUM_SIMULATIONS):
    ecoli70_200 = ecoli70_true.sample(200, seed=1 + (i*100))
    ecoli70_200.to_pandas().to_csv("data/ecoli70_" + str(i).zfill(3) + "_200.csv", index=False)
    ecoli70_2000 = ecoli70_true.sample(2000, seed=2 + (i*100))
    ecoli70_2000.to_pandas().to_csv("data/ecoli70_" + str(i).zfill(3) + "_2000.csv", index=False)
    ecoli70_10000 = ecoli70_true.sample(10000, seed=3 + (i*100))
    ecoli70_10000.to_pandas().to_csv("data/ecoli70_" +  str(i).zfill(3) + "_10000.csv", index=False)
    ecoli70_test = ecoli70_true.sample(1000, seed=4 + (i*100))
    ecoli70_test.to_pandas().to_csv("data/ecoli70_"+ str(i).zfill(3) + "_test.csv", index=False)

    magic_niab_200 = magic_niab_true.sample(200, seed=1 + (i*100))
    magic_niab_200.to_pandas().to_csv("data/magic_niab_" + str(i).zfill(3) + "_200.csv", index=False)
    magic_niab_2000 = magic_niab_true.sample(2000, seed=2 + (i*100))
    magic_niab_2000.to_pandas().to_csv("data/magic_niab_" + str(i).zfill(3) + "_2000.csv", index=False)
    magic_niab_10000 = magic_niab_true.sample(10000, seed=3 + (i*100))
    magic_niab_10000.to_pandas().to_csv("data/magic_niab_" + str(i).zfill(3) + "_10000.csv", index=False)
    magic_niab_test = magic_niab_true.sample(1000, seed=4 + (i*100))
    magic_niab_test.to_pandas().to_csv("data/magic_niab_"+ str(i).zfill(3) + "_test.csv", index=False)


    magic_irri_200 = magic_irri_true.sample(200, seed=1 + (i*100))
    magic_irri_200.to_pandas().to_csv("data/magic_irri_" + str(i).zfill(3) + "_200.csv", index=False)
    magic_irri_2000 = magic_irri_true.sample(2000, seed=2 + (i*100))
    magic_irri_2000.to_pandas().to_csv("data/magic_irri_" + str(i).zfill(3) + "_2000.csv", index=False)
    magic_irri_10000 = magic_irri_true.sample(10000, seed=3 + (i*100))
    magic_irri_10000.to_pandas().to_csv("data/magic_irri_" + str(i).zfill(3) + "_10000.csv", index=False)
    magic_irri_test = magic_irri_true.sample(1000, seed=4 + (i*100))
    magic_irri_test.to_pandas().to_csv("data/magic_irri_" + str(i).zfill(3) + "_test.csv", index=False)


    arth150_200 = arth150_true.sample(200, seed=1 + (i*100))
    arth150_200.to_pandas().to_csv("data/arth150_" + str(i).zfill(3) + "_200.csv", index=False)
    arth150_2000 = arth150_true.sample(2000, seed=2 + (i*100))
    arth150_2000.to_pandas().to_csv("data/arth150_" + str(i).zfill(3) + "_2000.csv", index=False)
    arth150_10000 = arth150_true.sample(10000, seed=3 + (i*100))
    arth150_10000.to_pandas().to_csv("data/arth150_" + str(i).zfill(3) + "_10000.csv", index=False)
    arth150_test = arth150_true.sample(1000, seed=4 + (i*100))
    arth150_test.to_pandas().to_csv("data/arth150_" + str(i).zfill(3) + "_test.csv", index=False)