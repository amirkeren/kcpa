from scipy.linalg import eigh
import numpy as np


def stepwise_kpca(k, n_components):
    # Centering the symmetric NxN kernel matrix.
    n = k.shape[0]
    one_n = np.ones((n, n)) / n
    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigvals, eigvecs = eigh(k - one_n.dot(k) - k.dot(one_n) + one_n.dot(k).dot(one_n))
    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    return np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))
