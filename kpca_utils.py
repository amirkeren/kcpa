from scipy.linalg import eigh
import numpy as np


def stepwise_kpca(k, n_components):
    n = k.shape[0]
    one_n = np.ones((n, n)) / n
    eigvals, eigvecs = eigh(k - one_n.dot(k) - k.dot(one_n) + one_n.dot(k).dot(one_n))
    return np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))
