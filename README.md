This repository contains the experiments for "Semiparametric Bayesian Networks."

There is a folder for each experiment type. `synthetic` for synthetic data experiments, `GaussianNetworks` for data sampled from Gaussian networks and `UCI data` for experiments from the UCI repository.

To run these experiments [`PyBNesian`](https://github.com/davenza/PyBNesian) is needed. 

To ensure the reproducibility of the experiments, we used the following experimental setup:

- The experiments were run using [this version](https://github.com/davenza/PyBNesian/tree/440cd9a94fa3ef5d8f3132b945bc521eef7e6406) of `PyBNesian`. We believe that the results should be reproducible with other versions.

- Ubuntu 16.04, Python 3.7.3, and compiled with gcc 9.3.0. We believe that the results should be reproducible for most configurations. Altough sometimes C++ compilers introduce optimizations and changes in the standard library implementation in new releases. These changes may subtly affect the results.

Organization
=================

All experiments contain scripts with the following structure:

- `experiments_helper.py`
- `train_[algorithm]_[model_type].py`
- `test_[algorithm]_[model_type].py`
- other code such as plotting scripts.


`experiments_helper.py` defines the parameters of the experiment at the start of the file. Also, it contains some auxiliary code used by the experiments.

The experiments are executed in two steps so the experiments can be paused/resumed easily. First, the models are learned from the training data and saved. For this, use the scripts `train_[algorithm]_[model_type].py`. This step can take quite some time, so the execution can be stopped at any moment for all training scripts. If the script is executed again, it will automatically detect the already learned models.

Then, the `test_[algorithm]_[model_type].py` scripts load the learned models and test them on unseen data. The results of this testing is saved in csv files or printed in the screen depending on the type of experiment.

Train scripts
-----------------------------

All the learned models are saved in a local folder called `models/`.

**Important: create a `models/` folder on each experiment folder.**

All the learned models (including all the iterations of the greedy hill-climbing algorithm) are saved.

The `[algorithm]` part can take the following values:

- `hc` for greedy hill-climbing.
- `pc` for PC.
 
The `[model_type]` part can take the following values:

- `graph`: This option is only available for PC algorithm. It learns the graph of the Bayesian network. **Important: when learning models with PC, this script should be run before learning the specific Bayesian networks described below.**
- `gbn` to learn Gaussian Networks.
- `kdebn` to learn kernel density estimation Bayesian networks.
- `spbn` to learn semiparametric Bayesian networks with linear Gaussian CPDs as starting node types.
- `spbn_ckde` to learn semiparametric Bayesian networks with conditional kernel density estimation CPDs as starting node types.

Test scripts
-----------------------

The structure of the test scripts is the same as in the train scripts. This process usually takes much less time than the training. For the UCI data experiment, the scripts save the results in CSV files called `results_[algorithm]_[model_type].csv`. For the other experiment types, the experiments are printed on screen.

Other scripts
-------------------

The remaining scripts contain useful utilities such as plotting, sampling of data or statistical tests.