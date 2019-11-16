from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel, linear_kernel, laplacian_kernel, sigmoid_kernel
import itertools


class KernelInitializer:
    def __init__(self, name, params):
        self.name = name
        self.current_params = []
        if name == "polynomial_kernel":
            self.params = itertools.product(params['gammas'], params['coefs'], params['degrees'])
        elif name == "rbf_kernel":
            self.params = itertools.product(params['gammas'])
        elif name == "laplacian_kernel":
            self.params = itertools.product(params['gammas'])
        elif name == "sigmoid_kernel":
            self.params = itertools.product(params['gammas'], params['coefs'])
        elif name == "linear_kernel":
            self.params = iter(list())
        else:
            raise NotImplementedError(name)

    def get_kernel_info(self):
        print(self.name, list(self.params), self.current_params)

    def get_kernel(self, x):
        try:
            self.current_params = next(self.params)
            if self.name == "polynomial_kernel":
                return polynomial_kernel(x, gamma=self.current_params[0], coef0=self.current_params[1], degree=self.current_params[2])
            if self.name == "rbf_kernel":
                return rbf_kernel(x, gamma=self.current_params[0])
            if self.name == "laplacian_kernel":
                return laplacian_kernel(x, gamma=self.current_params[0])
            if self.name == "sigmoid_kernel":
                return sigmoid_kernel(x, gamma=self.current_params[0], coef0=self.current_params[1])
            if self.name == "linear_kernel":
                return linear_kernel(x)
        except StopIteration:
            return None
