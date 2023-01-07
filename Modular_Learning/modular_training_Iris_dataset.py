import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import sklearn
import pandas as pd

data = load_iris()
data.target[[110, 65, 50]] # 0-50 arasi 0, 51-100 1, 101-149 2 bunlar indis
list(data.target_names)

data,target = sklearn.datasets.load_iris(return_X_y=True, as_frame=False)
print(data.shape)





"""



def k_mtrx(x0, x1):
    return np.matmul(tanh_norm(x0), tanh_norm(x1).T)

def map_input(x):
    return np.where(
         np.eye(len(x)) == 1,
            np.array(-float("inf"), dtype=np.float32),
            k_mtrx(x, x)
        )


def one_hot_encode(target, n_classes):
    
    if target.ndim > 1:
        target = np.squeeze(target)
    target_onehot = np.zeros((target.shape[0], n_classes))
    # target_onehot[range(target.size(0)), target] = 1
    # target_onehot[range(target.size(0)).long(), target.long()] = 1
    target_onehot[range(target.shape[0]), target.astype(int)] = 1
    return target_onehot

def get_ideal_k_mtrx(target1, target2, n_classes):
    
    k_min = -1.
    k_max = 1.
    if n_classes < 2:
        raise ValueError('You need at least 2 classes')

    if len(target1.shape) > 2:
        raise ValueError('target1 has too many dimensions')
    if len(target2.shape) > 2:
        raise ValueError('target2 has too many dimensions')

    if np.max(target1) + 1 > n_classes:
        raise ValueError('target1 has at least one invalid entry')
    if np.max(target2) + 1 > n_classes:
        raise ValueError('target2 has at least one invalid entry')

    target1_onehot, target2_onehot = \
        one_hot_encode(target1, n_classes).astype(np.float32), \
        one_hot_encode(target2, n_classes).astype(np.float32)
    ideal = np.matmul(target1_onehot, target2_onehot.T)
    if k_min != 0:
        min_mask = np.full(ideal.shape, k_min)
        ideal = np.where(ideal == 0, min_mask, ideal)

    if k_max != 1:
        max_mask = np.full_like(ideal, k_max)
        ideal = np.where(ideal == 1, max_mask, ideal)
    return ideal
"""