import numpy as np
np.random.seed(0)
import pandas as pd
import pathlib
import glob
from pybnesian import load
import experiments_helper

ecoli70_true = load('ecoli70.pickle')
magic_niab_true = load('magic_niab.pickle')
magic_irri_true = load('magic_irri.pickle')
arth150_true = load('arth150.pickle')

def compare_models(true_model, dataset_name, num_instances):
    ll = np.empty((experiments_helper.NUM_SIMULATIONS,))
    shd = np.empty((experiments_helper.NUM_SIMULATIONS,))
    hamming = np.empty((experiments_helper.NUM_SIMULATIONS,))

    for p in experiments_helper.PATIENCE:
        for i in range(experiments_helper.NUM_SIMULATIONS):
            train_df = pd.read_csv('data/' + dataset_name + "_" + str(i).zfill(3) + "_" + str(num_instances) + '.csv')
            test_df = pd.read_csv('data/' + dataset_name + "_" + str(i).zfill(3) + '_test.csv')
            
            spbn_ckde_lc_folder = 'models/' + dataset_name + '/' + str(i).zfill(3) + '/' + str(num_instances) + '/PC/SPBN/LinearCorrelation/' + str(p)
            all_models = sorted(glob.glob(spbn_ckde_lc_folder + '/*.pickle'))
            final_model = all_models[-1]
            spbn = load(final_model)
            spbn.fit(train_df)

            ll[i] = spbn.slogl(test_df)
            shd[i] = experiments_helper.shd(spbn, true_model)
            hamming[i] = experiments_helper.hamming(spbn, true_model)

        print("Mean SPBN LinearCorrelation results: ")
        print("Loglik: " + str(ll.mean()))
        print("SHD: " + str(shd.mean()))
        print("Hamming: " + str(hamming.mean()))
        print()

        for i in range(experiments_helper.NUM_SIMULATIONS):
            train_df = pd.read_csv('data/' + dataset_name + "_" + str(i).zfill(3) + "_" + str(num_instances) + '.csv')
            test_df = pd.read_csv('data/' + dataset_name + "_" + str(i).zfill(3) + '_test.csv')

            spbn_ckde_rcot_folder = 'models/' + dataset_name + '/' + str(i).zfill(3) + '/' + str(num_instances) + '/PC/SPBN/RCoT/' + str(p)
            all_models = sorted(glob.glob(spbn_ckde_rcot_folder + '/*.pickle'))
            final_model = all_models[-1]
            spbn = load(final_model)
            spbn.fit(train_df)

            ll[i] = spbn.slogl(test_df)
            shd[i] = experiments_helper.shd(spbn, true_model)
            hamming[i] = experiments_helper.hamming(spbn, true_model)

        print("Mean SPBN RCoT results: ")
        print("Loglik: " + str(ll.mean()))
        print("SHD: " + str(shd.mean()))
        print("Hamming: " + str(hamming.mean()))
        print()

for true_model, name in \
        zip([ecoli70_true, magic_niab_true, magic_irri_true, arth150_true],
            experiments_helper.DATASETS
        ):
    for i in experiments_helper.INSTANCES:
        compare_models(true_model, name, i)

