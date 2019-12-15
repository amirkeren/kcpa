from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import scale
import numpy as np
import random
import re

DEFAULT_RANDOM_DISTRIBUTION_SIZE = 10
DEFAULT_POLYNOMIAL_MULTIPLIER = 0.5
DEFAULT_POLYNOMIAL_DEGREE = 3
DEFAULT_COEFFICIENT_RANGE = [0.5, 1.5]
DEFAULT_R_RANGE = [1, 3]
DEFAULT_EXPONENT = 5
DEFAULT_SIGMOID_COEFFICIENT_RANGE = [-1, 0]
SCALE_NORMALIZATION = 'scale'
REGULAR_NORMALIZATION = 'regular'

np.seterr(divide='ignore', invalid='ignore')


class Kernel:
    def __init__(self, kernel_config, components_num):
        self.kernel_params = {}
        self.kernel_instances = {}
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
        distribution_size = kernel_config['distribution_size'] if 'distribution_size' in kernel_config else \
            DEFAULT_RANDOM_DISTRIBUTION_SIZE
        random_distribution = np.random.uniform(size=distribution_size)
        avg_random_distribution = np.mean(random_distribution)
        if kernel_name == 'linear':
            kernel_inner_params = {}
            kernel_instance = KernelPCA(n_components=self.n_components)
        elif kernel_name == 'polynomial':
            multiplier = kernel_config['poly_multiplier'] if 'poly_multiplier' in kernel_config else \
                DEFAULT_POLYNOMIAL_MULTIPLIER
            gamma = kernel_config['poly_gamma'] if 'poly_gamma' in kernel_config \
                else 1 / (multiplier * np.max(random_distribution))
            poly_coef = kernel_config['poly_coef'] if 'poly_coef' in kernel_config else DEFAULT_COEFFICIENT_RANGE
            coef0 = random.uniform(poly_coef[0], poly_coef[1]) * avg_random_distribution
            degree = kernel_config['poly_degree'] if 'poly_degree' in kernel_config else DEFAULT_POLYNOMIAL_DEGREE
            kernel_inner_params = {
                "gamma": gamma,
                "coef0": coef0,
                "degree": degree
            }
            kernel_instance = KernelPCA(n_components=self.n_components, kernel='poly', gamma=gamma, coef0=coef0,
                                        degree=degree)
        elif kernel_name == 'sigmoid':
            exp = kernel_config['sig_exp'] if 'sig_exp' in kernel_config else DEFAULT_EXPONENT
            gamma = kernel_config['sig_gamma'] if 'sig_gamma' in kernel_config else \
                1 / pow(avg_random_distribution, exp)
            sig_coef = kernel_config['sig_coef'] if 'sig_coef' in kernel_config else DEFAULT_SIGMOID_COEFFICIENT_RANGE
            coef0 = random.uniform(sig_coef[0], sig_coef[1])
            kernel_inner_params = {
                "gamma": gamma,
                "coef0": coef0
            }
            kernel_instance = KernelPCA(n_components=self.n_components, kernel='sigmoid', gamma=gamma, coef0=coef0)
        elif kernel_name == 'rbf':
            rbf_r = kernel_config['rbf_r'] if 'rbf_r' in kernel_config else DEFAULT_R_RANGE
            r = random.uniform(rbf_r[0], rbf_r[1])
            gamma = kernel_config['rbf_gamma'] if 'rbf_gamma' in kernel_config else 1 / pow(avg_random_distribution, r)
            kernel_inner_params = {
                "gamma": gamma
            }
            kernel_instance = KernelPCA(n_components=self.n_components, kernel='rbf', gamma=gamma)
        else:
            raise NotImplementedError('Unsupported kernel')
        self.kernel_params[kernel_name] = kernel_inner_params
        self.kernel_instances[kernel_name] = kernel_instance

    @staticmethod
    def _normalize_kernel(x, normalization_method=REGULAR_NORMALIZATION):
        if normalization_method == SCALE_NORMALIZATION:
            return scale(x, axis=0, with_mean=True, with_std=True, copy=True)
        return 2. * (x - np.min(x)) / np.ptp(x) - 1

    def calculate_kernel(self, x):
        kernel_calculation = np.zeros((x.shape[0], self.n_components))
        for kernel_function, kernel_instance in self.kernel_instances.items():
            temp_kernel_calculation = self._normalize_kernel(kernel_instance.fit_transform(x))
            if self.kernel_combine == '+':
                kernel_calculation += temp_kernel_calculation
            elif self.kernel_combine == '*':
                kernel_calculation *= temp_kernel_calculation
            else:
                kernel_calculation = temp_kernel_calculation
        return np.nan_to_num(self._normalize_kernel(kernel_calculation))

    def to_string(self):
        return str(self.kernel_params)
