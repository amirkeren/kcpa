from os import listdir, makedirs
from os.path import isfile, join, exists
from kernels_manager import get_kernel
from classifiers_manager import get_classifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import localtime, strftime, ctime
import pandas as pd
import multiprocessing as mp
import json
import itertools
import copy

DATASETS_FOLDER = 'datasets'
RESULTS_FOLDER = 'results'


def run_experiment(output, dataset, experiment, kernels, classifier_config, components_num):
    dataset_name = dataset[0]
    df = dataset[1]
    x = df.iloc[:, :-1]
    components_num = components_num if isinstance(components_num, int) else x.shape[1] // 2
    y = df.iloc[:, -1]
    results = {}
    i = 0
    while i < len(kernels):
        kernel_config = kernels[i]
        kernel_instances = kernel_config['instances'] if 'instances' in kernel_config else 10
        if kernel_instances > 1:
            duplicated_kernels = []
            for _ in range(kernel_instances - 1):
                duplicate_kernel = copy.deepcopy(kernel_config)
                duplicate_kernel['instances'] = 1
                duplicated_kernels.append(duplicate_kernel)
            kernels.extend(duplicated_kernels)
        kernel_name = kernel_config['name']
        kernel, kernel_params = get_kernel(x, kernel_config, components_num)
        kernel_config['run_params'] = kernel_params
        X_train, X_test, y_train, y_test = train_test_split(kernel, y, test_size=0.3, random_state=42)
        clf = get_classifier(classifier_config)
        clf = clf.fit(X_train, y_train)
        results[kernel_name] = clf.predict(X_test)
        i += 1
    df = pd.DataFrame.from_dict(results)
    output.put({
        "experiment": experiment,
        "kernels": kernels,
        "components": components_num,
        "classifier": classifier_config,
        "dataset": dataset_name,
        "accuracy": metrics.accuracy_score(y_test, df.mode(axis=1).iloc[:, 0])
    })


def write_results(dataset_name, data):
    if not exists(RESULTS_FOLDER):
        makedirs(RESULTS_FOLDER)
    current_time = strftime('%Y%m%d-%H%M%S', localtime())
    filename = RESULTS_FOLDER + '/' + current_time + '-' + dataset_name + '.json'
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def main():
    with open('experiments.json') as json_data_file:
        experiments = json.load(json_data_file)
        datasets = [f for f in listdir(DATASETS_FOLDER) if isfile(join(DATASETS_FOLDER, f))]
        datasets.sort()
        output = mp.Queue()
        for dataset in [(dataset, pd.read_csv(join(DATASETS_FOLDER, dataset), header=None)) for dataset in datasets]:
            print(ctime(), 'Starting to run experiments on dataset', dataset[0])
            processes = []
            for experiment_name, experiment_params in experiments.items():
                components = experiment_params['components'] if 'components' in experiment_params else [10, '0.5d']
                classifiers_list = experiment_params['classifiers'] if 'classifiers' in experiment_params else [
                  {
                    "name": "decision_tree"
                  },
                  {
                    "name": "nearest_neighbors"
                  }
                ]
                for experiment_config in itertools.product(classifiers_list, components):
                    classifiers = experiment_config[0]
                    n_components = experiment_config[1]
                    p = mp.Process(target=run_experiment,
                                   args=(output, dataset, experiment_name, experiment_params['kernels'], classifiers,
                                         n_components))
                    processes.append(p)
            for p in processes:
                p.start()
            write_results(dataset[0], [output.get() for p in processes])
            print(ctime(), 'Finished running experiments on dataset', dataset[0])


if __name__ == '__main__':
    main()
