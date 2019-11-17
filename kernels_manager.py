from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel, linear_kernel, laplacian_kernel, sigmoid_kernel
import functools


class KernelsManager:
    def __init__(self, x):
        self.x = x

    def get_kernel(self, kernel):
        kernel_name = kernel['name']
        kernel_params = kernel['params']
        if '+' in kernel_name:
            return functools.reduce(lambda a, b: a + b, [self.get_kernel(
                {"name": kernel, "params": kernel_params}) for kernel in kernel_name.split('+')])
        if '*' in kernel_name:
            return functools.reduce(lambda a, b: a * b, [self.get_kernel(
                {"name": kernel, "params": kernel_params}) for kernel in kernel_name.split('*')])
        if kernel_name == "polynomial":
            return polynomial_kernel(self.x, gamma=kernel_params['gamma'], coef0=kernel_params['coef'],
                                     degree=kernel_params['degree'])
        if kernel_name == "rbf":
            return rbf_kernel(self.x, gamma=kernel_params['gamma'])
        if kernel_name == "laplacian":
            return laplacian_kernel(self.x, gamma=kernel_params['gamma'])
        if kernel_name == "sigmoid":
            return sigmoid_kernel(self.x, gamma=kernel_params['gamma'], coef0=kernel_params['coef'])
        if kernel_name == "linear":
            return linear_kernel(self.x)
