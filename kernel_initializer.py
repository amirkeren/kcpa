from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel, linear_kernel, laplacian_kernel, sigmoid_kernel
import itertools
import json


class KernelInitializer:
    def __init__(self, x, kernels):
        kernel_name = kernels[0]
        with open('kernels.json') as json_data_file:
            self.kernels_config = json.load(json_data_file)
        if kernel_name not in self.kernels_config.keys():
            raise NotImplementedError(kernel_name)
        self.kernel_name = kernel_name
        self.current_params = []
        self.x = x
        self.params = itertools.product(*list(self.kernels_config[kernel_name].values()))

    def get_kernel(self):
        try:
            self.current_params = next(self.params)
            if self.kernel_name == "polynomial":
                return polynomial_kernel(self.x, gamma=self.current_params[0], coef0=self.current_params[1], degree=self.current_params[2])
            if self.kernel_name == "rbf":
                return rbf_kernel(self.x, gamma=self.current_params[0])
            if self.kernel_name == "laplacian":
                return laplacian_kernel(self.x, gamma=self.current_params[0])
            if self.kernel_name == "sigmoid":
                return sigmoid_kernel(self.x, gamma=self.current_params[0], coef0=self.current_params[1])
            if self.kernel_name == "linear":
                return linear_kernel(self.x)
        except StopIteration:
            return None
