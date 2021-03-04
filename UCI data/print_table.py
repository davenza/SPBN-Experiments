import os
import experiments_helper



def print_datasets():
    datasets = experiments_helper.find_crossvalidation_datasets()

    for dataset in datasets:
        basefolder = os.path.basename(os.path.dirname(dataset))
        x = experiments_helper.validate_dataset(dataset, experiments_helper.TRAINING_FOLDS)
        if x is None:
            continue
        else:
            dataset, _ = x
        print("{} & {} & {}\\\\".format(basefolder, dataset.shape[0], dataset.shape[1]))



if __name__ == "__main__":
    print_datasets()