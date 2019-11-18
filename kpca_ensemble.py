from os import listdir
from os.path import isfile, join
from kernels_manager import get_kernel
from classifiers_manager import get_classifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from time import localtime, strftime
import pandas as pd
import multiprocessing as mp
import json
import itertools

output = mp.Queue()


def run_experiment(dataset, experiment, kernels, classifier_config, components_num):
    dataset_name = dataset[0]
    df = dataset[1]
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    results = {}
    for kernel_config in kernels:
        kernel_name = kernel_config['name']
        kernel = get_kernel(x, kernel_config, components_num)
        X_train, X_test, y_train, y_test = train_test_split(kernel, y, test_size=0.3, random_state=42)
        clf = get_classifier(classifier_config)
        clf = clf.fit(X_train, y_train)
        results[kernel_name] = clf.predict(X_test)
    df = pd.DataFrame.from_dict(results)
    output.put({
        "experiment": experiment,
        "kernels": kernels,
        "components": n_components,
        "classifier": classifier_config,
        "dataset": dataset_name,
        "accuracy": metrics.accuracy_score(y_test, df.mode(axis=1))
    })


def write_results(dataset_name, data):
    current_time = strftime("%Y%m%d-%H%M%S", localtime())
    filename = 'results/' + current_time + '-' + dataset_name + '.json'
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


with open('experiments.json') as json_data_file:
    experiments = json.load(json_data_file)
    datasets = [f for f in listdir('datasets') if isfile(join('datasets', f))]
    datasets.sort()
    for dataset in [(dataset, pd.read_csv(join('datasets', dataset), header=None)) for dataset in datasets][1:]:
        processes = []
        for experiment_name, experiment_params in experiments.items():
            for experiment_config in itertools.product(experiment_params['classifiers'], experiment_params['components']):
                classifiers = experiment_config[0]
                n_components = experiment_config[1]
                p = mp.Process(target=run_experiment,
                               args=(dataset, experiment_name, experiment_params['kernels'], classifiers, n_components))
                processes.append(p)
        for p in processes:
            p.start()
        write_results(dataset[0], [output.get() for p in processes])
