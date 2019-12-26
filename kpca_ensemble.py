from email import encoders
from email.mime import text
from email.mime.application import MIMEApplication
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from os import listdir, makedirs
from os.path import isfile, join, exists, basename
from kernel import Kernel, Normalization
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

SEND_DETAILED_EMAIL = True
DATASETS_FOLDER = 'datasets'
RESULTS_FOLDER = 'results'
ACCURACY_FLOATING_POINT = 5
DEFALUT_CROSS_VALIDATION_FOR_BASELINE = 10
KERNELS_TO_CHOOSE = 10

DEFAULT_NUMBER_OF_KERNELS = [10, 25]
DEFAULT_NUMBER_OF_COMPONENTS = ['0.9d', '0.75d', '0.5d']
DEFAULT_CROSS_VALIDATION = [10, 2]
DEFAULT_NORMALIZATION_METHODS = [Normalization.STANDARD, Normalization.ABSOLUTE, Normalization.NEGATIVE,
                                 Normalization.SCALE]


def send_email(user, pwd, recipient, subject, body, file):
    print('Sending summary email')
    to = recipient if type(recipient) is list else [recipient]
    msg = MIMEMultipart()
    msg['From'] = user
    msg['To'] = COMMASPACE.join(to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    attachment = open(file, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % file)
    msg.attach(part)
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(user, pwd)
    server.sendmail(user, to, msg.as_string())
    server.quit()
    print('Summary email sent')


def build_experiment_key(experiment_name, classifier, components=None, folds=None, kernels_num=None, normalization=None,
                         kernels=None):
    key = experiment_name + '-' + classifier
    if components:
        key += '-' + str(components)
    if folds:
        key += '-' + str(folds)
    if kernels_num:
        key += '-' + str(kernels_num)
    if normalization:
        key += '-' + str(normalization)
    if kernels:
        key += '-[' + (','.join([kernel.to_string() for kernel in kernels])) + ']'
    return key


def write_results_to_csv(dataframe):
    if not exists(RESULTS_FOLDER):
        makedirs(RESULTS_FOLDER)
    current_time = strftime('%Y%m%d-%H%M%S', localtime())
    filename = RESULTS_FOLDER + '/results-' + current_time + '.csv'
    dataframe.to_csv(filename)
    return filename


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
        normalization_method = Normalization[experiment_params['normalization']] \
            if 'normalization' in experiment_params else DEFAULT_NORMALIZATION_METHODS
        for _ in itertools.product(classifiers_list, components, cross_validation, ensemble_size, normalization_method):
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
        dataframe = dataset[1]
        dataframe = dataframe.fillna(dataframe.mean())
        X = dataframe.iloc[:, :-1]
        y = dataframe.iloc[:, -1]
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
            normalization_method = Normalization[experiment_params['normalization']] \
                if 'normalization' in experiment_params else DEFAULT_NORMALIZATION_METHODS
            for experiment_config in itertools.product(classifiers_list, components, cross_validation, ensemble_size,
                                                       normalization_method):
                classifier_config = experiment_config[0]
                components_num = experiment_config[1]
                folds = experiment_config[2]
                kernels_num = experiment_config[3]
                normalization = experiment_config[4]
                components_num = components_num if isinstance(components_num, int) else \
                    round(X.shape[1] * float(components_num[:-1]))
                kernels = [Kernel(experiment_params['kernel'], components_num, normalization) for _ in
                           itertools.repeat(None, kernels_num)]
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
                                          folds, kernels_num, normalization, kernels), accuracy))
                count += 1
                print(ctime(), '{0:.1%}'.format(float(count) / total_number_of_experiments), dataset_name,
                      build_experiment_key(experiment_name, classifier_config['name'], components_num, folds,
                                           kernels_num, normalization))
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
        result_df = pd.DataFrame.from_dict(data, orient='index', columns=column_names)
        return result_df, write_results_to_csv(df)


def summarize_results(results_df):
    summary_results = {}
    for i in range(1, len(CLASSIFIERS) + 1):
        summary_results[results_df.columns[i].split('-')[1]] = {
            "baseline_accuracy": round(results_df.iloc[:, i].mean(), ACCURACY_FLOATING_POINT),
            "baseline_results": results_df.iloc[:, i],
            "experiments_accuracy": []
        }
    for i in range(len(CLASSIFIERS) + 1, len(results_df.columns)):
        summary_results[results_df.columns[i].split('-')[1]]['experiments_accuracy'].append({
            "experiment": results_df.columns[i],
            "experiment_results": results_df.iloc[:, i],
            "accuracy": round(results_df.iloc[:, i].mean(), ACCURACY_FLOATING_POINT)
        })
    for key, value in summary_results.items():
        best_experiment = {'accuracy': -1}
        for experiment in value['experiments_accuracy']:
            if experiment['accuracy'] > best_experiment['accuracy']:
                best_experiment = experiment
        summary_results[key]['best_experiment'] = best_experiment
        del summary_results[key]['experiments_accuracy']
    return summary_results


def run_statistical_analysis(results_df):
    print('Run statistical analysis on results')
    summarized_results = summarize_results(results_df)
    results_string = ''
    for key, value in summarized_results.items():
        baseline = value['baseline_results']
        experiment = value['best_experiment']['experiment_results']
        baseline_mean = value['baseline_accuracy']
        experiment_mean = value['best_experiment']['accuracy']
        results_string += key + '\n'
        if baseline_mean > experiment_mean:
            results_string += 'Baseline wins: ' + str(baseline_mean) + ' > ' + str(experiment_mean) + '\n'
        else:
            results_string += 'Experiment wins: ' + str(experiment_mean) + ' > ' + str(baseline_mean) + '\n'
        results_string += 'Best experiment: ' + str(value['best_experiment']['experiment']) + '\n'
        t, p = stats.ttest_ind(baseline, experiment)
        results_string += 'T-Test: t = ' + str(round(t, ACCURACY_FLOATING_POINT)) + 'p = ' + \
                          str(round(p, ACCURACY_FLOATING_POINT)) + '\n'
        stat, p = stats.wilcoxon(baseline, experiment)
        results_string += 'Wilcoxon: s = ' + str(stat) + 'p = ' + str(round(p, ACCURACY_FLOATING_POINT)) + '\n'
        results_string += '\n'
    return results_string


if __name__ == '__main__':
    input_file = None
    df = None
    send_summary_email = True
    for arg in sys.argv[1:]:
        input_file = arg
    if input_file:
        send_summary_email = False
        input_file = 'results/' + input_file
        if isfile(input_file):
            print('Results file found')
            df = pd.read_csv(input_file)
        else:
            print('Results file', input_file, 'not found')
            df, input_file = get_experiments_results()
    else:
        df, input_file = get_experiments_results()
    stat_results = run_statistical_analysis(df)
    print(stat_results)
    config = configparser.RawConfigParser()
    config.read('ConfigFile.properties')
    if send_summary_email:
        send_email(config.get('EmailSection', 'email.user'), config.get('EmailSection', 'email.password'),
                   'ak091283@gmail.com', 'Finished Running', stat_results if SEND_DETAILED_EMAIL else '', input_file)
