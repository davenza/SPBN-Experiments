import numpy as np
np.random.seed(0)
import pandas as pd
import pathlib
import glob
from pybnesian import load
import experiments_helper
from generate_dataset_spbn import slogl_model

true_model = load('true_model.pickle')

def compare_models(true_model, num_instances):
    ll = np.empty((experiments_helper.NUM_SIMULATIONS,))
    shd = np.empty((experiments_helper.NUM_SIMULATIONS,))
    hamming = np.empty((experiments_helper.NUM_SIMULATIONS,))
    hamming_type = np.empty((experiments_helper.NUM_SIMULATIONS,))

    for i in range(experiments_helper.NUM_SIMULATIONS):
        test_df = pd.read_csv('data/synthetic_' + str(i).zfill(3) + '_test.csv')
        ll[i] = slogl_model(test_df)

    print("True model loglik: " + str(ll.mean()))

    for p in experiments_helper.PATIENCE:
        for i in range(experiments_helper.NUM_SIMULATIONS):
            train_df = pd.read_csv('data/synthetic_' + str(i).zfill(3) + "_" + str(num_instances) + '.csv')
            test_df = pd.read_csv('data/synthetic_' + str(i).zfill(3) + '_test.csv')

            result_folder = 'models/' +  str(i).zfill(3) + '/' + str(num_instances) + '/HillClimbing/SPBN_CKDE/' + str(p)
            pathlib.Path(result_folder).mkdir(parents=True, exist_ok=True)

            all_models = sorted(glob.glob(result_folder + '/*.pickle'))
            final_model = load(all_models[-1])
            final_model.fit(train_df)

            ll[i] = final_model.slogl(test_df)
            shd[i] = experiments_helper.shd(final_model, true_model)
            hamming[i] = experiments_helper.hamming(final_model, true_model)
            hamming_type[i] = experiments_helper.hamming_type(final_model, true_model)

        print("Loglik, p " + str(p) + ": " + str(ll.mean()))
        print("SHD, p " + str(p) + ": " + str(shd.mean()))
        print("Hamming, p " + str(p) + ": " + str(hamming.mean()))
        print("Hamming type, p " + str(p) + ": " + str(hamming_type.mean()))

        print()

for i in experiments_helper.INSTANCES:
    compare_models(true_model, i)

