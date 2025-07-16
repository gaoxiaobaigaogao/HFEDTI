
import numpy as np
from sklearn.model_selection import KFold
import random

def get_kfold_data(i, datasets, k=10):

    fold_size = len(datasets) // k

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:]
        trainset = datasets[0:val_start]

    return trainset, validset

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def get_stratified_kfold_data(i, pos_samples, neg_samples, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    pos_folds = list(kf.split(pos_samples))
    neg_folds = list(kf.split(neg_samples))

    pos_train_idx, pos_val_idx = pos_folds[i]
    neg_train_idx, neg_val_idx = neg_folds[i]

    trainset = [pos_samples[idx] for idx in pos_train_idx] + [neg_samples[idx] for idx in neg_train_idx]
    validset = [pos_samples[idx] for idx in pos_val_idx] + [neg_samples[idx] for idx in neg_val_idx]

    random.shuffle(trainset)
    random.shuffle(validset)
    return trainset, validset
