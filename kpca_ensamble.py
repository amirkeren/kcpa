from experiments_manager import ExperimentsManager
from time import localtime, strftime
import json


def write_results(data):
    current_time = strftime("%Y%m%d-%H%M%S", localtime())
    filename = 'results/' + current_time + '-' + data['experiment_name'] + '-' + data['dataset'] + '.json'
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

    experiments_manager = ExperimentsManager()
    experiment = experiments_manager.get_next_experiment()
    experiments_manager.run(experiment)


# for experiment_name, experiment_params in experiments.items():
#     kernels = experiment_params['kernels']
#     join_by_functions = None if 'joinByFunctions' not in experiment_params else experiment_params['joinByFunctions']
#     for num_components in experiment_params['components']:
#         for dataset in datasets:
#             kernel_manager = KernelsManager(X, kernels, join_by_functions)
#             while True:
#                 kernel, running_configuration = kernel_manager.get_kernel()
#                 if kernel is None:
#                     break
#                 X_pc = stepwise_kpca(kernel, n_components=num_components)
#                 results_json = {
#                     'components': num_components,
#                     'dataset': dataset,
#                     'experiment_name': experiment_name,
#                     'running_configuration': running_configuration,
#                     'results': 'TODO'
#                 }
#                 write_results(results_json)
