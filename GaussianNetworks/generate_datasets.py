import numpy as np
np.random.seed(0)
from pgmpy.models import LinearGaussianBayesianNetwork


ecoli70_true = LinearGaussianBayesianNetwork.load_model('ecoli70.pkl')

ecoli70_200 = ecoli70_true.sample_dataset(200)
ecoli70_200.to_csv("ecoli70_200.csv", index=False)
ecoli70_2000 = ecoli70_true.sample_dataset(2000)
ecoli70_2000.to_csv("ecoli70_2000.csv", index=False)
ecoli70_10000 = ecoli70_true.sample_dataset(10000)
ecoli70_10000.to_csv("ecoli70_10000.csv", index=False)
ecoli70_test = ecoli70_true.sample_dataset(1000)
ecoli70_test.to_csv("ecoli70_test.csv", index=False)

magic_niab_true = LinearGaussianBayesianNetwork.load_model('magic_niab.pkl')

magic_niab_200 = magic_niab_true.sample_dataset(200)
magic_niab_200.to_csv("magic_niab_200.csv", index=False)
magic_niab_2000 = magic_niab_true.sample_dataset(2000)
magic_niab_2000.to_csv("magic_niab_2000.csv", index=False)
magic_niab_10000 = magic_niab_true.sample_dataset(10000)
magic_niab_10000.to_csv("magic_niab_10000.csv", index=False)
magic_niab_test = magic_niab_true.sample_dataset(1000)
magic_niab_test.to_csv("magic_niab_test.csv", index=False)

magic_irri_true = LinearGaussianBayesianNetwork.load_model('magic_irri.pkl')

magic_irri_200 = magic_irri_true.sample_dataset(200)
magic_irri_200.to_csv("magic_irri_200.csv", index=False)
magic_irri_2000 = magic_irri_true.sample_dataset(2000)
magic_irri_2000.to_csv("magic_irri_2000.csv", index=False)
magic_irri_10000 = magic_irri_true.sample_dataset(10000)
magic_irri_10000.to_csv("magic_irri_10000.csv", index=False)
magic_irri_test = magic_irri_true.sample_dataset(1000)
magic_irri_test.to_csv("magic_irri_test.csv", index=False)

arth150_true = LinearGaussianBayesianNetwork.load_model('arth150.pkl')

arth150_200 = arth150_true.sample_dataset(200)
arth150_200.to_csv("arth150_200.csv", index=False)
arth150_2000 = arth150_true.sample_dataset(2000)
arth150_2000.to_csv("arth150_2000.csv", index=False)
arth150_10000 = arth150_true.sample_dataset(10000)
arth150_10000.to_csv("arth150_10000.csv", index=False)
arth150_test = arth150_true.sample_dataset(1000)
arth150_test.to_csv("arth150_test.csv", index=False)