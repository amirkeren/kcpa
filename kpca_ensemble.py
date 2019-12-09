from os import listdir, makedirs
from os.path import isfile, join, exists
from kernels_manager import get_kernel, stepwise_kpca
from classifiers_manager import get_classifier, CLASSIFIERS
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn import metrics
from time import localtime, strftime, ctime
import pandas as pd
import numpy as np
import multiprocessing as mp
import json
import itertools
import copy
import smtplib

DATASETS_FOLDER = 'datasets'
RESULTS_FOLDER = 'results'
DATAFRAME_COLUNMS = ['Dataset', 'Experiment', 'Classifier', 'Components', 'Folds', 'Accuracy', 'Kernels']
ACCURACY_FLOATING_POINT = 5
DEFALUT_CROSS_VALIDATION_FOR_BASELINE = 10

DEFAULT_NUMBER_OF_KERNELS = [10, 'best-25']
DEFAULT_NUMBER_OF_COMPONENTS = [10, '0.5d']
DEFAULT_CROSS_VALIDATION = [10, 2]


def send_email(user, pwd, recipient, subject, body):
    to = recipient if type(recipient) is list else [recipient]
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s""" % (user, ", ".join(to), subject, body)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(user, pwd)
        server.sendmail(user, to, message)
        server.close()
        print('Email sent successfully')
    except:
        print('Failed to send mail')


def build_experiment_key(experiment_name, classifier, components=None, folds=None, kernels=None):
    key = experiment_name + '-' + classifier
    if components:
        key += '-' + str(components)
    if folds:
        key += '-' + str(folds)
    if kernels:
        key += '-' + str(kernels)
    return key


def write_results_to_csv(dataframe):
    if not exists(RESULTS_FOLDER):
        makedirs(RESULTS_FOLDER)
    current_time = strftime('%Y%m%d-%H%M%S', localtime())
    dataframe.to_csv(RESULTS_FOLDER + '/results-' + current_time + '.csv')


def get_total_number_of_experiments(experiments):
    count = 0
    for experiment_name, experiment_params in experiments.items():
        components = experiment_params['components'] if 'components' in experiment_params \
            else DEFAULT_NUMBER_OF_COMPONENTS
        cross_validation = experiment_params['cross_validation'] if 'cross_validation' in experiment_params \
            else DEFAULT_CROSS_VALIDATION
        classifiers_list = experiment_params['classifiers'] if 'classifiers' in experiment_params \
            else CLASSIFIERS
        for _ in itertools.product(classifiers_list, components, cross_validation):
            count += 1
    return count


def run_baseline(dataset_name, X, y):
    intermediate_results = {}
    experiment_name = 'baseline'
    X = X.to_numpy()
    y = y.to_numpy()
    for classifier_config in CLASSIFIERS:
        clf = get_classifier(classifier_config)
        accuracy = round(np.mean(cross_val_score(clf, X, y, cv=DEFALUT_CROSS_VALIDATION_FOR_BASELINE)),
                         ACCURACY_FLOATING_POINT)
        intermediate_results.setdefault(dataset_name, []).append((
            build_experiment_key(experiment_name, classifier_config['name']), accuracy))
    return intermediate_results


def run_experiments(output, dataset, experiments):
    dataset_name = dataset[0]
    print(ctime(), 'Starting to run experiments on dataset', dataset_name)
    total_number_of_experiments = get_total_number_of_experiments(experiments)
    df = dataset[1]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    intermediate_results = run_baseline(dataset_name, X, y)
    count = 0
    for experiment_name, experiment_params in experiments.items():
        components = experiment_params['components'] if 'components' in experiment_params \
            else DEFAULT_NUMBER_OF_COMPONENTS
        cross_validation = experiment_params['cross_validation'] if 'cross_validation' in experiment_params \
            else DEFAULT_CROSS_VALIDATION
        classifiers_list = experiment_params['classifiers'] if 'classifiers' in experiment_params \
            else CLASSIFIERS
        for experiment_config in itertools.product(classifiers_list, components, cross_validation):
            classifier_config = experiment_config[0]
            components_num = experiment_config[1]
            cross_validation = experiment_config[2]
            components_num = components_num if isinstance(components_num, int) else X.shape[1] // 2
            results = {}
            i = 0
            kernels = copy.deepcopy(experiment_params['kernels'])
            while i < len(kernels):
                kernel_config = kernels[i]
                kernel_instances = kernel_config['instances'] if 'instances' in kernel_config \
                    else DEFAULT_NUMBER_OF_KERNELS
                if kernel_instances > 1:
                    duplicated_kernels = []
                    for _ in range(kernel_instances - 1):
                        duplicate_kernel = copy.deepcopy(kernel_config)
                        duplicate_kernel['instances'] = 1
                        duplicated_kernels.append(duplicate_kernel)
                    kernels.extend(duplicated_kernels)
                i += 1
            accuracies = []
            if cross_validation == 2:  # 5x2 cross validation
                if len(kernels) < 5:
                    raise Exception('Cannot perform 5x2 cross validation with an ensemble of less than 5 kernels')
                random_states = [0, 7, 9, 23, 42]
                for i in range(5):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                                        random_state=random_states[i])
                    kernel_config = kernels[i]
                    kernel_name = kernel_config['name'] + '_' + str(i)
                    results[kernel_name] = train_predict(X_train, X_test, y_train, classifier_config, kernel_config,
                                                         components_num)
                df = pd.DataFrame.from_dict(results)
                accuracies.append(metrics.accuracy_score(y_test, df.mode(axis=1).iloc[:, 0]))
            else:
                kf = KFold(n_splits=cross_validation)
                for train_index, test_index in kf.split(X):
                    results = {}
                    X_train, X_test = X.values[train_index], X.values[test_index]
                    y_train, y_test = y.values[train_index], y.values[test_index]
                    for i, kernel_config in enumerate(kernels):
                        kernel_name = kernel_config['name'] + '_' + str(i)
                        results[kernel_name] = train_predict(X_train, X_test, y_train, classifier_config, kernel_config,
                                                             components_num)
                    df = pd.DataFrame.from_dict(results)
                    accuracies.append(metrics.accuracy_score(y_test, df.mode(axis=1).iloc[:, 0]))
            accuracy = round(np.asarray(accuracies).mean(), ACCURACY_FLOATING_POINT)
            intermediate_results.setdefault(dataset_name, []).append(
                (build_experiment_key(experiment_name, classifier_config['name'], components_num,
                                      cross_validation, kernels), accuracy))
            count += 1
            print(ctime(), '{0:.1%}'.format(float(count) / total_number_of_experiments),
                  build_experiment_key(experiment_name, classifier_config['name'], components_num, cross_validation))
    print(ctime(), 'Finished running experiments on dataset', dataset_name)
    output.put(intermediate_results)


def train_predict(X_train, X_test, y_train, classifier_config, kernel_config, components_num):
    kernel_calculation, kernel_params = get_kernel(X_train, kernel_config)
    kernel_config['run_params'] = kernel_params
    train_kernel = stepwise_kpca(kernel_calculation, components_num)
    test_kernel = stepwise_kpca(get_kernel(X_test, kernel_config, kernel_params)[0], components_num)
    clf = get_classifier(classifier_config)
    clf = clf.fit(train_kernel, y_train)
    return clf.predict(test_kernel)


def main():
    with open('experiments.json') as json_data_file:
        print(ctime(), 'Starting to run experiments')
        experiments = json.load(json_data_file)
        datasets = [f for f in listdir(DATASETS_FOLDER) if isfile(join(DATASETS_FOLDER, f))]
        datasets.sort()
        output = mp.Queue()
        processes = []
        for dataset in [(dataset, pd.read_csv(join(DATASETS_FOLDER, dataset), header=None)) for dataset in datasets]:
            p = mp.Process(target=run_experiments, args=(output, dataset, experiments))
            processes.append(p)
        for p in processes:
            p.start()
        results = [output.get() for _ in processes]
        print(ctime(), 'Finished running all experiments')
        data = {}
        column_names = []
        first_iteration_only = True
        for process_results in results:
            for dataset_name, results in process_results.items():
                accuracies = []
                for result in results:
                    if first_iteration_only:
                        column_names.append(result[0])
                    accuracies.append(result[1])
                first_iteration_only = False
                data[dataset_name] = accuracies
        df = pd.DataFrame.from_dict(data, orient='index', columns=column_names)
        write_results_to_csv(df)
        send_email('kagglemailsender', 'Amir!1@2#3$4', 'ak091283@gmail.com', 'Finished Running', df)


if __name__ == '__main__':
    main()
