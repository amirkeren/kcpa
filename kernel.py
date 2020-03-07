from kernel_pca import KernelPCA
from normalization import Normalization
import numpy as np
import random
import re

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh

DEFAULT_RANDOM_DISTRIBUTION_SIZE = 10
DEFAULT_POLYNOMIAL_DEGREE = 2
DEFAULT_GAMMA_RANGE = [-1, 1]
DEFAULT_R_RANGE = [1, 3]
DEFAULT_EXPONENT = 5
DEFAULT_SIGMOID_COEFFICIENT_RANGE = [-1, 0]

np.seterr(divide='ignore', invalid='ignore')

kernel_to_normalization = {
    'polynomial': Normalization.STANDARD,
    'linear': Normalization.STANDARD,
    'sigmoid': Normalization.STANDARD,
    'rbf': Normalization.ABSOLUTE
}


def _rbf_kernel_pca(x, gamma, n_components):
    sq_dists = pdist(x, 'sqeuclidean')
    mat_sq_dists = squareform(sq_dists)
    K = exp(-gamma * mat_sq_dists)
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    _, eigvecs = eigh(K)
    eigvecs = eigvecs[:, ::-1]
    return np.column_stack([eigvecs[:, i] for i in range(n_components)])


class Kernel:
    def __init__(self, kernel_config, components_num, avg_euclidean_distances, max_euclidean_distances):
        self.kernel_params = {}
        self.kernel_instances = {}
        self.avg_euclidean_distances = avg_euclidean_distances
        self.max_euclidean_distances = max_euclidean_distances
        self.kernel_combine = None
        self.n_components = components_num
        self.kernel_name = kernel_config['name']
        if '+' in self.kernel_name:
            self.kernel_combine = '+'
        elif '*' in self.kernel_name:
            self.kernel_combine = '*'
        for kernel_name in re.split('[*+]', kernel_config['name']):
            self._generate_kernel(kernel_config, kernel_name)

    def _generate_kernel(self, kernel_config, kernel_name):
        if kernel_name == 'linear':
            kernel_inner_params = {}
            kernel_instance = KernelPCA(n_components=self.n_components)
        elif kernel_name == 'polynomial':
            gamma_range = kernel_config['poly_gamma'] if 'poly_gamma' in kernel_config else DEFAULT_GAMMA_RANGE
            gamma = random.uniform(gamma_range[0], gamma_range[1])
            degree = kernel_config['poly_degree'] if 'poly_degree' in kernel_config else DEFAULT_POLYNOMIAL_DEGREE
            kernel_inner_params = {
                "gamma": gamma,
                "degree": degree
            }
            kernel_instance = KernelPCA(n_components=self.n_components, kernel='poly', gamma=gamma, degree=degree)
        elif kernel_name == 'sigmoid':
            distribution_size = kernel_config['distribution_size'] if 'distribution_size' in kernel_config else \
                DEFAULT_RANDOM_DISTRIBUTION_SIZE
            random_distribution = np.random.uniform(size=distribution_size)
            avg_random_distribution = np.mean(random_distribution)
            exp = kernel_config['sig_exp'] if 'sig_exp' in kernel_config else DEFAULT_EXPONENT
            gamma = kernel_config['sig_gamma'] if 'sig_gamma' in kernel_config else 1 / (avg_random_distribution ** exp)
            sig_coef = kernel_config['sig_coef'] if 'sig_coef' in kernel_config else DEFAULT_SIGMOID_COEFFICIENT_RANGE
            coef0 = random.uniform(sig_coef[0], sig_coef[1])
            kernel_inner_params = {
                "gamma": gamma,
                "coef0": coef0
            }
            kernel_instance = KernelPCA(n_components=self.n_components, kernel='sigmoid', coef0=coef0)
        elif kernel_name == 'rbf':
            rbf_r = kernel_config['rbf_r'] if 'rbf_r' in kernel_config else DEFAULT_R_RANGE
            r = np.random.uniform(rbf_r[0], rbf_r[1])
            gamma = kernel_config['rbf_gamma'] if 'rbf_gamma' in kernel_config \
                else 1 / (self.avg_euclidean_distances ** r)
            kernel_inner_params = {
                "gamma": gamma
            }
            kernel_instance = KernelPCA(n_components=self.n_components, kernel='rbf', gamma=gamma)
        else:
            raise NotImplementedError('Unsupported kernel')
        self.kernel_params[kernel_name] = kernel_inner_params
        self.kernel_instances[kernel_name] = kernel_instance

    def calculate_kernel(self, x, is_test=False):
        for kernel_function, kernel_instance in self.kernel_instances.items():
            if is_test:
                transformed = kernel_instance.transform(x)
            else:
                transformed = kernel_instance.fit_transform(x)
        return transformed

    def to_string(self):
        return str(self.kernel_params)
