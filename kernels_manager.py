from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel, linear_kernel, laplacian_kernel, sigmoid_kernel, euclidean_distances
from scipy.linalg import eigh
import numpy as np
import functools
import warnings
import random


def get_kernel(x, kernel_config, n_components):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        euclid_distances = euclidean_distances(x) if kernel_config['name'] != "linear" else None
        return _stepwise_kpca(_get_kernel(x, kernel_config, euclid_distances), n_components)


def _get_kernel(x, kernel_config, euclid_distances):
    kernel_name = kernel_config['name']
    kernel_params = kernel_config['params']
    if '+' in kernel_name:
        return functools.reduce(lambda a, b: a + b, [_get_kernel(x,
            {"name": kernel, "params": kernel_params}, euclid_distances) for kernel in kernel_name.split('+')])
    if '*' in kernel_name:
        return functools.reduce(lambda a, b: a * b, [_get_kernel(x,
            {"name": kernel, "params": kernel_params}, euclid_distances) for kernel in kernel_name.split('*')])
    if kernel_name == "linear":
        return linear_kernel(x)
    avg_euclid_distance = np.average(euclid_distances)
    if kernel_name == "polynomial":
        multiplier = kernel_params['multiplier'] if 'multiplier' in kernel_params else 0.5
        gamma = kernel_params['gamma'] if 'gamma' in kernel_params else 1 / (multiplier * np.max(euclid_distances))
        coef0 = random.uniform(kernel_params['coef'][0], kernel_params['coef'][1]) * avg_euclid_distance
        return polynomial_kernel(x, gamma=gamma, coef0=coef0, degree=kernel_params['degree'])
    if kernel_name == "rbf":
        r = random.uniform(kernel_params['r'][0], kernel_params['r'][1])
        gamma = kernel_params['gamma'] if 'gamma' in kernel_params else 1 / pow(avg_euclid_distance, r)
        return rbf_kernel(x, gamma=gamma)
    if kernel_name == "laplacian":
        exp = kernel_params['exp'] if 'exp' in kernel_params else 5
        gamma = kernel_params['gamma'] if 'gamma' in kernel_params else 1 / pow(avg_euclid_distance, exp)
        return laplacian_kernel(x, gamma=gamma)
    if kernel_name == "sigmoid":
        exp = kernel_params['exp'] if 'exp' in kernel_params else 5
        gamma = kernel_params['gamma'] if 'gamma' in kernel_params else 1 / pow(avg_euclid_distance, exp)
        coef0 = random.uniform(kernel_params['coef'][0], kernel_params['coef'][1])
        return sigmoid_kernel(x, gamma=gamma, coef0=coef0)


def _stepwise_kpca(k, n_components):
    n = k.shape[0]
    one_n = np.ones((n, n)) / n
    _, eigvecs = eigh(k - one_n.dot(k) - k.dot(one_n) + one_n.dot(k).dot(one_n))
    return np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))
