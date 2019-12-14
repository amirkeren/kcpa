from sklearn.decomposition import KernelPCA
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


class Kernel:
    def __init__(self, kernel_config):
        self.kernel_params = {}
        self.kernel_combine = None
        self.kernel_name = kernel_config['name']
        if '+' in self.kernel_name:
            self.kernel_combine = '+'
        elif '*' in self.kernel_name:
            self.kernel_combine = '*'
        for kernel_name in re.split('[*+]', kernel_config['name']):
            self._generate_kernel(kernel_config, kernel_name, self.kernel_params)

    def _generate_kernel(self, kernel_config, kernel_name, kernel_params):
        kernel_inner_params = {}
        distribution_size = kernel_config['distribution_size'] if 'distribution_size' in kernel_config else \
            DEFAULT_RANDOM_DISTRIBUTION_SIZE
        random_distribution = np.random.uniform(size=distribution_size)
        avg_random_distribution = np.mean(random_distribution)
        if kernel_name == 'linear':
            pass
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
        elif kernel_name == 'rbf':
            rbf_r = kernel_config['rbf_r'] if 'rbf_r' in kernel_config else DEFAULT_R_RANGE
            r = random.uniform(rbf_r[0], rbf_r[1])
            gamma = kernel_config['rbf_gamma'] if 'rbf_gamma' in kernel_config else 1 / pow(avg_random_distribution, r)
            kernel_inner_params = {
                "gamma": gamma
            }
        elif kernel_name == 'laplacian':
            exp = kernel_config['lap_exp'] if 'lap_exp' in kernel_config else DEFAULT_EXPONENT
            gamma = kernel_config['lap_gamma'] if 'lap_gamma' in kernel_config else \
                1 / pow(avg_random_distribution, exp)
            kernel_inner_params = {
                "gamma": gamma
            }
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
        else:
            raise NotImplementedError('Unsupported kernel')
        kernel_params[kernel_name] = kernel_inner_params

    def calculate_kernel(self, x, n_components):
        kernel_calculation = np.zeros((x.shape[0], n_components))
        for kernel_function, kernel_params in self.kernel_params.items():
            if kernel_function == 'linear':
                kernel = KernelPCA(n_components=n_components)
            elif kernel_function == 'polynomial':
                kernel = KernelPCA(n_components=n_components, kernel='polynomial', gamma=kernel_params['gamma'],
                                   coef0=kernel_params['coef0'], degree=kernel_params['degree'])
            elif kernel_function == 'rbf':
                kernel = KernelPCA(n_components=n_components, kernel='rbf', gamma=kernel_params['gamma'])
            elif kernel_function == 'laplacian':
                kernel = KernelPCA(n_components=n_components, kernel='laplacian', gamma=kernel_params['gamma'])
            elif kernel_function == 'sigmoid':
                kernel = KernelPCA(n_components=n_components, kernel='sigmoid', gamma=kernel_params['gamma'],
                                   coef0=kernel_params['coef0'])
            else:
                raise NotImplementedError('Unsupported kernel')
            temp_kernel_calculation = kernel.fit_transform(x)
            if self.kernel_combine == '+':
                kernel_calculation += temp_kernel_calculation
            elif self.kernel_combine == '*':
                kernel_calculation *= temp_kernel_calculation
            else:
                kernel_calculation = temp_kernel_calculation
        return kernel_calculation

    def get_kernel(self):
        return self.kernel_name, self.kernel_params

    def to_string(self):
        return str(self.kernel_params)
