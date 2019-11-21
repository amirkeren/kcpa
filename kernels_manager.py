from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel, linear_kernel, laplacian_kernel, sigmoid_kernel
from scipy.linalg import eigh
import numpy as np
import functools
import warnings


def get_kernel(x, kernel_config, n_components):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _stepwise_kpca(_get_kernel(x, kernel_config), n_components)


def _get_kernel(x, kernel_config):
    kernel_name = kernel_config['name']
    kernel_params = kernel_config['params']
    if '+' in kernel_name:
        return functools.reduce(lambda a, b: a + b, [_get_kernel(x,
            {"name": kernel, "params": kernel_params}) for kernel in kernel_name.split('+')])
    if '*' in kernel_name:
        return functools.reduce(lambda a, b: a * b, [_get_kernel(x,
            {"name": kernel, "params": kernel_params}) for kernel in kernel_name.split('*')])
    if kernel_name == "polynomial":
        return polynomial_kernel(x, gamma=kernel_params['gamma'], coef0=kernel_params['coef'],
                                 degree=kernel_params['degree'])
    if kernel_name == "rbf":
        return rbf_kernel(x, gamma=kernel_params['gamma'])
    if kernel_name == "laplacian":
        return laplacian_kernel(x, gamma=kernel_params['gamma'])
    if kernel_name == "sigmoid":
        return sigmoid_kernel(x, gamma=kernel_params['gamma'], coef0=kernel_params['coef'])
    if kernel_name == "linear":
        return linear_kernel(x)


def _stepwise_kpca(k, n_components):
    n = k.shape[0]
    one_n = np.ones((n, n)) / n
    _, eigvecs = eigh(k - one_n.dot(k) - k.dot(one_n) + one_n.dot(k).dot(one_n))
    return np.column_stack((eigvecs[:, -i] for i in range(1, n_components + 1)))
