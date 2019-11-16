from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel, linear_kernel, laplacian_kernel, sigmoid_kernel
import itertools
import json
import functools


class KernelsManager:
    def __init__(self, x, kernels, join_by_functions):
        if len(kernels) > 1 and join_by_functions is None:
            raise Exception('invalid experiment - more than one kernel defined and no method to join by')
        with open('kernels.json') as json_data_file:
            self.kernels_config = json.load(json_data_file)
        running_configurations = []
        for kernel_name in kernels:
            if kernel_name not in self.kernels_config.keys():
                raise Exception(kernel_name)
            running_configurations.append(list(itertools.product([kernel_name],
                                                                 *list(self.kernels_config[kernel_name].values()))))
        if join_by_functions:
            running_configurations.append(join_by_functions)
        self.x = x
        self.num_kernels = len(kernels)
        self.running_configurations = itertools.product(*running_configurations)

    def get_kernel(self):
        try:
            running_configuration = next(self.running_configurations)
            if self.num_kernels == 1:
                return self._get_kernel(running_configuration[0])
            kernels = []
            for kernel in running_configuration[:-1]:
                kernels.append(self._get_kernel(kernel))
            if running_configuration[-1] == "add":
                return functools.reduce(lambda a, b: a + b, kernels)
            elif running_configuration[-1] == "mul":
                return functools.reduce(lambda a, b: a * b, kernels)
        except StopIteration:
            return None

    def _get_kernel(self, kernel):
        kernel_name = kernel[0]
        if kernel_name == "polynomial":
            return polynomial_kernel(self.x, gamma=kernel[1], coef0=kernel[2], degree=kernel[3])
        if kernel_name == "rbf":
            return rbf_kernel(self.x, gamma=kernel[1])
        if kernel_name == "laplacian":
            return laplacian_kernel(self.x, kernel[1])
        if kernel_name == "sigmoid":
            return sigmoid_kernel(self.x, gamma=kernel[1], coef0=kernel[2])
        if kernel_name == "linear":
            return linear_kernel(self.x)
