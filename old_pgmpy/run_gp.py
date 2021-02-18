import os
import experiments_helper
import multiprocessing as mp
import pathlib
from joblib import dump, load

from sklearn.model_selection import KFold
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic, ExpSineSquared, DotProduct
from sklearn.gaussian_process import GaussianProcessRegressor


def run_gp_components(train_data, kernels, kernels_names, result_folder, idx_fold):
    for k, names in zip(kernels, kernels_names):
        fold_folder = result_folder + '/GP/' + names + '/' + str(idx_fold)
        pathlib.Path(fold_folder).mkdir(parents=True, exist_ok=True)

        if os.path.exists(fold_folder + '/end.lock'):
            continue

        gp = GaussianProcessRegressor(k, n_restarts_optimizer=2, normalize_y=True, copy_X_train=True, random_state=0)
        # FIXME: GP solo funciona con regression
        gp.fit(train_data)
        dump(gp, fold_folder + '/model.pkl')
        with open(fold_folder + '/end.lock', 'w') as f:
            pass


def train_crossvalidation_file(file, kernels, kernels_names):
    print("Training " + str(file))

    x = experiments_helper.validate_dataset(file, [2, 3, 5, 10])
    if x is None:
        return
    else:
        dataset, result_folder = x

    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    if not os.path.exists(result_folder + '/GP'):
        os.mkdir(result_folder + '/GP')

    for (idx_fold, (train_indices, test_indices)) in enumerate(KFold(10, shuffle=True, random_state=0).split(dataset)):
        run_gp_components(dataset.iloc[train_indices,:], kernels, kernels_names, result_folder, idx_fold)

    # with mp.Pool(processes=10) as p:
    #     p.starmap(run_gp_components, [(dataset.iloc[train_indices,:], kernels, kernels_names, result_folder, idx_fold)
    #                                          for (idx_fold, (train_indices, test_indices)) in
    #                                          enumerate(KFold(10, shuffle=True, random_state=0).split(dataset))]
    #               )

def train_crossvalidation():
    files = experiments_helper.find_crossvalidation_datasets()

    kernels = [1.0*RBF(),
               1.0*RBF() + 1.0*WhiteKernel(),
               1.0*Matern() + 1.0*WhiteKernel(),
               1.0*RationalQuadratic() + 1.0*WhiteKernel(),
               1.0*ExpSineSquared() + 1.0*WhiteKernel(),
               1.0*DotProduct() + 1.0*WhiteKernel()]
    kernels_names = ["OnlyRBF", "RBF", "Matern", "RationalQuadratic", "ExpSineSquared", "DotProduct"]

    for file in files:
        train_crossvalidation_file(file, kernels, kernels_names)

if __name__ == '__main__':
    train_crossvalidation()
    # test_crossvalidation()