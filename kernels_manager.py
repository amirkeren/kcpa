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
            log_configuration = {
                'joinBy': running_configuration[-1]
            }
            log_kernels_configuration = []
            for kernel in running_configuration[:-1]:
                temp_kernel, inner_configuration = self._get_kernel(kernel)
                kernels.append(temp_kernel)
                log_kernels_configuration.append(inner_configuration)
            log_configuration['kernels'] = log_kernels_configuration
            if running_configuration[-1] == "add":
                return functools.reduce(lambda a, b: a + b, kernels), log_configuration
            elif running_configuration[-1] == "mul":
                return functools.reduce(lambda a, b: a * b, kernels), log_configuration
        except StopIteration:
            return None, None

    def _get_kernel(self, kernel_config):
        kernel_name = kernel_config[0]
        configuration = {}
        if len(kernel_config) >= 2:
            configuration['gamma'] = kernel_config[1]
        if len(kernel_config) >= 3:
            configuration['coef0'] = kernel_config[2]
        if len(kernel_config) >= 4:
            configuration['degree'] = kernel_config[3]
        configuration['kernel'] = kernel_name
        if kernel_name == "polynomial":
            kernel = polynomial_kernel(self.x, gamma=kernel_config[1], coef0=kernel_config[2], degree=kernel_config[3])
        elif kernel_name == "rbf":
            kernel = rbf_kernel(self.x, gamma=kernel_config[1])
        elif kernel_name == "laplacian":
            kernel = laplacian_kernel(self.x, kernel_config[1])
        elif kernel_name == "sigmoid":
            kernel = sigmoid_kernel(self.x, gamma=kernel_config[1], coef0=kernel_config[2])
        elif kernel_name == "linear":
            kernel = linear_kernel(self.x)
        return kernel, configuration
