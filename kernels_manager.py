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
        kernel_calculation, kernel_run_params = _get_kernel(x, kernel_config, euclid_distances)
        return _stepwise_kpca(kernel_calculation, n_components), kernel_run_params


def _get_kernel(x, kernel_config, euclid_distances):
    kernel_name = kernel_config['name']
    kernel_params = kernel_config['params']
    if '+' in kernel_name:
        return functools.reduce(lambda a, b: (a[0] + b[0], {**a[1], **b[1]}), [_get_kernel(x,
            {"name": kernel, "params": kernel_params}, euclid_distances) for kernel in kernel_name.split('+')])
    if '*' in kernel_name:
        return functools.reduce(lambda a, b: (a[0] * b[0], {**a[1], **b[1]}), [_get_kernel(x,
            {"name": kernel, "params": kernel_params}, euclid_distances) for kernel in kernel_name.split('*')])
    if kernel_name == "linear":
        kernel_run_params = {}
        kernel_calculation = linear_kernel(x)
    else:
        avg_euclid_distance = np.average(euclid_distances)
        if kernel_name == "polynomial":
            multiplier = kernel_params['poly_multiplier'] if 'poly_multiplier' in kernel_params else 0.5
            gamma = kernel_params['poly_gamma'] if 'poly_gamma' in kernel_params \
                else 1 / (multiplier * np.max(euclid_distances))
            coef0 = random.uniform(kernel_params['poly_coef'][0], kernel_params['poly_coef'][1]) * avg_euclid_distance
            kernel_run_params = {
                "poly_multiplier": multiplier,
                "poly_gamma": gamma,
                "poly_coef0": coef0
            }
            kernel_calculation = polynomial_kernel(x, gamma=gamma, coef0=coef0, degree=kernel_params['poly_degree'])
        if kernel_name == "rbf":
            r = random.uniform(kernel_params['rbf_r'][0], kernel_params['rbf_r'][1])
            gamma = kernel_params['rbf_gamma'] if 'rbf_gamma' in kernel_params else 1 / pow(avg_euclid_distance, r)
            kernel_run_params = {
                "rbf_r": r,
                "rbf_gamma": gamma
            }
            kernel_calculation = rbf_kernel(x, gamma=gamma)
        if kernel_name == "laplacian":
            exp = kernel_params['lap_exp'] if 'lap_exp' in kernel_params else 5
            gamma = kernel_params['lap_gamma'] if 'lap_gamma' in kernel_params else 1 / pow(avg_euclid_distance, exp)
            kernel_run_params = {
                "lap_exp": exp,
                "lap_gamma": gamma
            }
            kernel_calculation = laplacian_kernel(x, gamma=gamma)
        if kernel_name == "sigmoid":
            exp = kernel_params['sig_exp'] if 'sig_exp' in kernel_params else 5
            gamma = kernel_params['sig_gamma'] if 'sig_gamma' in kernel_params else 1 / pow(avg_euclid_distance, exp)
            coef0 = random.uniform(kernel_params['sig_coef'][0], kernel_params['sig_coef'][1])
            kernel_run_params = {
                "sig_exp": exp,
                "sig_gamma": gamma,
                "sig_coef0": coef0
            }
            kernel_calculation = sigmoid_kernel(x, gamma=gamma, coef0=coef0)
    return kernel_calculation, kernel_run_params


def _stepwise_kpca(k, n_components):
    n = k.shape[0]
    one_n = np.ones((n, n)) / n
    _, eigvecs = eigh(k - one_n.dot(k) - k.dot(one_n) + one_n.dot(k).dot(one_n))
    return np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))
