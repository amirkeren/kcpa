from scipy.linalg import eigh
from sklearn.datasets import make_moons
from experiments_manager import ExperimentsManager
from kernel_initializer import KernelInitializer

import numpy as np
import matplotlib.pyplot as plt


def stepwise_kpca(k, n_components):
    # Centering the symmetric NxN kernel matrix.
    n = k.shape[0]
    one_n = np.ones((n, n)) / n
    # Obtaining eigenvalues in descending order with corresponding
    # eigenvectors from the symmetric matrix.
    eigvals, eigvecs = eigh(k - one_n.dot(k) - k.dot(one_n) + one_n.dot(k).dot(one_n))
    # Obtaining the i eigenvectors that corresponds to the i highest eigenvalues.
    return np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))


X, y = make_moons(n_samples=100, random_state=123)

config = ExperimentsManager().print_config()
for kernel_config in config['kernels']:
    kernel_initializer = KernelInitializer(kernel_config['name'], kernel_config['params'])
    while True:
        kernel = kernel_initializer.get_kernel(X)
        if kernel is None:
            break
        X_pc = stepwise_kpca(kernel, n_components=2)
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pc[y == 0, 0], X_pc[y == 0, 1], color='red', alpha=0.5)
        plt.scatter(X_pc[y == 1, 0], X_pc[y == 1, 1], color='blue', alpha=0.5)
        plt.title('First 2 principal components after RBF Kernel PCA')
        plt.text(-0.18, 0.18, 'gamma = 15', fontsize=12)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.show()
