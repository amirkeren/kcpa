from os import listdir, makedirs
from os.path import isfile, join, exists
from kernel import Kernel
from classifiers_manager import get_classifier, CLASSIFIERS
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedKFold
from sklearn import metrics
from time import localtime, strftime, ctime
from scipy import stats
import pandas as pd
import numpy as np
import multiprocessing as mp
import sys
import json
import itertools
import smtplib
import configparser


DATASETS_FOLDER = 'datasets'
RESULTS_FOLDER = 'results'
ACCURACY_FLOATING_POINT = 5
DEFALUT_CROSS_VALIDATION_FOR_BASELINE = 10
KERNELS_TO_CHOOSE = 10
DEFAULT_NUMBER_OF_KERNELS = [10]  # [10, 25]
DEFAULT_NUMBER_OF_COMPONENTS = ['0.5d']  # ['0.75d', '0.5d']
DEFAULT_CROSS_VALIDATION = [10]  # [10, 2]


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


def build_experiment_key(experiment_name, classifier, components=None, folds=None, kernels_num=None, kernels=None):
    key = experiment_name + '-' + classifier
    if components:
        key += '-' + str(components)
    if folds:
        key += '-' + str(folds)
    if kernels_num:
        key += '-' + str(kernels_num)
    if kernels:
        key += '-[' + (','.join([kernel.to_string() for kernel in kernels])) + ']'
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
        ensemble_size = experiment_params['ensemble_size'] if 'ensemble_size' in experiment_params \
            else DEFAULT_NUMBER_OF_KERNELS
        for _ in itertools.product(classifiers_list, components, cross_validation, ensemble_size):
            count += 1
    return count


def run_baseline(dataset_name, X, y):
    intermediate_results = {}
    experiment_name = 'baseline'
    for classifier_config in CLASSIFIERS:
        clf = get_classifier(classifier_config)
        accuracy = round(np.mean(cross_val_score(clf, X, y, cv=DEFALUT_CROSS_VALIDATION_FOR_BASELINE)),
                         ACCURACY_FLOATING_POINT)
        intermediate_results.setdefault(dataset_name, []).append((
            build_experiment_key(experiment_name, classifier_config['name']), accuracy))
    return intermediate_results


def choose_best_kernels(kernels, X, y, clf):
    kernels_heap = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    for kernel in kernels:
        embedded_train = kernel.calculate_kernel(X_train)
        embedded_test = kernel.calculate_kernel(X_test)
        clf = clf.fit(embedded_train, y_train)
        kernels_heap.append((kernel, metrics.accuracy_score(y_test, clf.predict(embedded_test))))
    return [tup[0] for tup in sorted(kernels_heap, key=lambda tup: tup[1])[-KERNELS_TO_CHOOSE:]]


def run_experiments(output, dataset, experiments):
    try:
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
            ensemble_size = experiment_params['ensemble_size'] if 'ensemble_size' in experiment_params \
                else DEFAULT_NUMBER_OF_KERNELS
            for experiment_config in itertools.product(classifiers_list, components, cross_validation, ensemble_size):
                classifier_config = experiment_config[0]
                components_num = experiment_config[1]
                folds = experiment_config[2]
                kernels_num = experiment_config[3]
                components_num = components_num if isinstance(components_num, int) else \
                    round(X.shape[1] * float(components_num[:-1]))
                kernels = [Kernel(experiment_params['kernel'], components_num) for _ in itertools.repeat(None,
                                                                                                         kernels_num)]
                accuracies = []
                clf = get_classifier(classifier_config)
                if len(kernels) > KERNELS_TO_CHOOSE:
                    kernels = choose_best_kernels(kernels, X, y, clf)
                n_repeats = 5 if folds == 2 else 1  # if folds == 2 => 5x2 cross validation
                rkf = RepeatedKFold(n_splits=folds, n_repeats=n_repeats, random_state=0)
                for train_index, test_index in rkf.split(X):
                    results = {}
                    X_train, X_test = X.values[train_index], X.values[test_index]
                    y_train, y_test = y.values[train_index], y.values[test_index]
                    for kernel in kernels:
                        embedded_train = kernel.calculate_kernel(X_train)
                        embedded_test = kernel.calculate_kernel(X_test)
                        clf = clf.fit(embedded_train, y_train)
                        results[kernel] = clf.predict(embedded_test)
                    results_df = pd.DataFrame.from_dict(results)
                    accuracies.append(metrics.accuracy_score(y_test, results_df.mode(axis=1).iloc[:, 0]))
                accuracy = round(np.asarray(accuracies).mean(), ACCURACY_FLOATING_POINT)
                intermediate_results.setdefault(dataset_name, []).append(
                    (build_experiment_key(experiment_name, classifier_config['name'], components_num,
                                          folds, kernels_num, kernels), accuracy))
                count += 1
                print(ctime(), '{0:.1%}'.format(float(count) / total_number_of_experiments), dataset_name,
                      build_experiment_key(experiment_name, classifier_config['name'], components_num, folds,
                                           kernels_num))
        print(ctime(), 'Finished running experiments on dataset', dataset_name)
        output.put(intermediate_results)
    except Exception as e:
        print(e)


def get_experiments_results():
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
        return df


def run_statistical_analysis(results_df):
    a = results_df.iloc[:, 1]
    b = results_df.iloc[:, 2]
    c = results_df.iloc[:, 3]
    t2, p2 = stats.ttest_ind(a, b)
    print("t = " + str(t2))
    print("p = " + str(p2))
    stat, p = stats.f_oneway(a, b, c)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')


if __name__ == '__main__':
    input_file = None
    df = None
    for arg in sys.argv[1:]:
        input_file = arg
    if input_file:
        df = pd.read_csv('results/' + input_file)
    else:
        df = get_experiments_results()
    run_statistical_analysis(df)
    config = configparser.RawConfigParser()
    config.read('ConfigFile.properties')
    # send_email(config.get('EmailSection', 'email.user'), config.get('EmailSection', 'email.password'),
    #            'ak091283@gmail.com', 'Finished Running', df)
