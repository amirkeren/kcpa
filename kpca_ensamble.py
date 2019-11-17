from experiments_manager import ExperimentsManager
from kernels_manager import KernelsManager
from kpca_utils import stepwise_kpca
from time import localtime, strftime
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import json


def write_results(data):
    current_time = strftime("%Y%m%d-%H%M%S", localtime())
    filename = 'results/' + current_time + '-' + data['experiment_name'] + '-' + data['dataset'] + '.json'
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


# experiments = ExperimentsManager.get_experiments()
datasets = ExperimentsManager.get_datasets()
clf = DecisionTreeClassifier(random_state=42)

for dataset, df in datasets:
    print(dataset)
    print(cross_val_score(clf, df.iloc[:, :-1], df.iloc[:, -1], cv=2))

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
