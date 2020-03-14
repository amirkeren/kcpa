from distutils import util
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from enum import Enum
from os import listdir, makedirs
from os.path import isfile, join, exists
from kernel import Kernel
from classifiers_manager import get_classifier, CLASSIFIERS
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from normalization import normalize, Normalization
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
import math


class CandidationMethod(Enum):
    BEST = 1
    MIXED = 2
    NONE = 3


RUN_ON_LARGE_DATASETS = False
SEND_EMAIL = False
DATASETS_FOLDER = 'datasets'
LARGE_DATASETS_FOLDER = 'large_datasets'
RESULTS_FOLDER = 'results'
ACCURACY_FLOATING_POINT = 5
KERNELS_TO_CHOOSE = 11
DEFAULT_NUMBER_OF_FOLDS = 10  # 2
DEFAULT_CANDIDATION_METHOD = CandidationMethod.BEST
DEFAULT_NORMALIZATION_METHOD = Normalization.STANDARD
DEFAULT_NUMBER_OF_KERNELS = [20]
DEFAULT_NUMBER_OF_COMPONENTS = ['0.5d']


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


def build_experiment_key(experiment_name, classifier, components=None, folds=None, kernels_num=None,
                         candidation_method=None, kernels=None):
    key = experiment_name + '-' + classifier
    if components:
        key += '-' + str(components)
    if folds:
        key += '-' + str(folds)
    if kernels_num:
        key += '-' + str(kernels_num)
    if candidation_method:
        key += '-' + str(candidation_method)
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


def get_experiment_parameters(experiment_params):
    classifiers_list = experiment_params['classifiers'] if 'classifiers' in experiment_params \
        else CLASSIFIERS
    components = experiment_params['components'] if 'components' in experiment_params \
        else DEFAULT_NUMBER_OF_COMPONENTS
    ensemble_size = experiment_params['ensemble_size'] if 'ensemble_size' in experiment_params \
        else DEFAULT_NUMBER_OF_KERNELS
    return classifiers_list, components, ensemble_size


def get_total_number_of_experiments(experiments):
    count = 0
    for _, experiment_params in experiments.items():
        for _ in itertools.product(*get_experiment_parameters(experiment_params)):
            count += 1
    return count


def run_baseline(dataset_name, X, y, splits):
    intermediate_results = {}
    experiment_name = 'baseline'
    for classifier_config in CLASSIFIERS:
        splits, splits_copy = itertools.tee(splits)
        clf = get_classifier(classifier_config)
        accuracy = round(np.mean(cross_val_score(clf, X, y, cv=splits_copy)), ACCURACY_FLOATING_POINT)
        intermediate_results.setdefault(dataset_name, []).append((
            build_experiment_key(experiment_name, classifier_config['name']), accuracy))
    return intermediate_results


def evaluate_all_kernels(kernels, X, y, classifier_config, splits):
    kernels_heap = []
    for kernel in kernels:
        accuracies = []
        splits, splits_copy = itertools.tee(splits)
        for train_index, test_index in splits_copy:
            X_train, X_test = X.values[train_index], X.values[test_index]
            y_train, y_test = y.values[train_index], y.values[test_index]
            embedded_train = kernel.calculate_kernel(X_train)
            embedded_test = kernel.calculate_kernel(X_test, is_test=True)
            clf = get_classifier(classifier_config)
            clf.fit(embedded_train, y_train)
            accuracies.append(metrics.accuracy_score(y_test, clf.predict(embedded_test)))
        kernels_heap.append((kernel, round(np.mean(accuracies), ACCURACY_FLOATING_POINT)))
    return sorted(kernels_heap, key=lambda tup: tup[1])


def choose_best_kernels(kernels_and_evaluations, method):
    if method == CandidationMethod.BEST:
        return [tup[0] for tup in kernels_and_evaluations[-KERNELS_TO_CHOOSE:]]
    if method == CandidationMethod.MIXED:
        top = math.ceil(KERNELS_TO_CHOOSE / 2)
        rest = KERNELS_TO_CHOOSE - top
        kernels_result = [tup[0] for tup in kernels_and_evaluations[:rest]]
        kernels_result.extend([tup[0] for tup in kernels_and_evaluations[-top:]])
        return kernels_result


def run_experiments(output, dataset, experiments):
    try:
        dataset_name = dataset[0].split('\\')[1]
        print(ctime(), 'Starting to run experiments on dataset', dataset_name)
        total_number_of_experiments = get_total_number_of_experiments(experiments)
        dataframe = dataset[1]
        dataframe = dataframe.fillna(dataframe.mean())
        X = dataframe.iloc[:, :-1]
        euclid_distances = euclidean_distances(X, X)
        avg_euclid_distances = np.average(euclid_distances)
        max_euclid_distances = np.max(euclid_distances)
        y = dataframe.iloc[:, -1]
        splits = RepeatedStratifiedKFold(n_splits=DEFAULT_NUMBER_OF_FOLDS,
                                         n_repeats=5 if DEFAULT_NUMBER_OF_FOLDS == 2 else 1, random_state=0).split(X, y)
        splits, splits_copy = itertools.tee(splits)
        intermediate_results = run_baseline(dataset_name, X, y, splits_copy)
        count = 0
        for experiment_name, experiment_params in experiments.items():
            print(ctime(), 'Starting to run experiments', experiment_name, 'on', dataset_name)
            for experiment_config in itertools.product(*get_experiment_parameters(experiment_params)):
                classifier_config = experiment_config[0]
                if bool(util.strtobool(classifier_config['ensemble'])):
                    count += 1
                    intermediate_results.setdefault(dataset_name, []).append(
                        (build_experiment_key(experiment_name, classifier_config['name'], components_str,
                                              DEFAULT_NUMBER_OF_FOLDS, kernels_num, DEFAULT_CANDIDATION_METHOD,
                                              kernels), -1))
                    break
                components_str = experiment_config[1]
                kernels_num = experiment_config[2]
                components_num = components_str if isinstance(components_str, int) else \
                    round(X.shape[1] * float(components_str[:-1]))
                kernels = [Kernel(experiment_params['kernel'], components_num, avg_euclid_distances,
                                  max_euclid_distances) for _ in itertools.repeat(None, kernels_num)]
                splits, splits_copy = itertools.tee(splits)
                if len(kernels) > KERNELS_TO_CHOOSE and DEFAULT_CANDIDATION_METHOD != CandidationMethod.NONE:
                    kernels = choose_best_kernels(evaluate_all_kernels(kernels, X, y, classifier_config, splits_copy),
                                                  DEFAULT_CANDIDATION_METHOD)
                accuracies = []
                for train_index, test_index in splits_copy:
                    results = {}
                    X_train, X_test = X.values[train_index], X.values[test_index]
                    y_train, y_test = y.values[train_index], y.values[test_index]
                    for kernel in kernels:
                        embedded_train = kernel.calculate_kernel(X_train)
                        embedded_test = kernel.calculate_kernel(X_test, is_test=True)
                        clf = get_classifier(classifier_config)
                        clf.fit(embedded_train, y_train)
                        results[kernel] = clf.predict(embedded_test)
                    results_df = pd.DataFrame.from_dict(results)
                    ensemble_vote = results_df.mode(axis=1).iloc[:, 0]
                    accuracies.append(metrics.accuracy_score(y_test, ensemble_vote))
                accuracy = round(np.asarray(accuracies).mean(), ACCURACY_FLOATING_POINT)
                intermediate_results.setdefault(dataset_name, []).append(
                    (build_experiment_key(experiment_name, classifier_config['name'], components_str,
                                          DEFAULT_NUMBER_OF_FOLDS, kernels_num, DEFAULT_CANDIDATION_METHOD,
                                          kernels), accuracy))
                count += 1
                print(ctime(), '{0:.1%}'.format(float(count) / total_number_of_experiments), dataset_name,
                      build_experiment_key(experiment_name, classifier_config['name'], components_str,
                                           DEFAULT_NUMBER_OF_FOLDS, kernels_num, DEFAULT_CANDIDATION_METHOD))
        print(ctime(), 'Finished running experiments on dataset', dataset_name)
        output.put(intermediate_results)
    except Exception as e:
        print(e)


def get_datasets():
    datasets = []
    if RUN_ON_LARGE_DATASETS:
        datasets.extend([LARGE_DATASETS_FOLDER + '\\' + f for f in listdir(LARGE_DATASETS_FOLDER)
                         if isfile(join(LARGE_DATASETS_FOLDER, f))])
    datasets.extend([DATASETS_FOLDER + '\\' + f for f in listdir(DATASETS_FOLDER) if
                     isfile(join(DATASETS_FOLDER, f))])
    return [(dataset, pd.read_csv(dataset, header=None)) for dataset in datasets]


def get_experiments_results():
    with open('experiments.json') as json_data_file:
        print(ctime(), 'Starting to run experiments')
        experiments = json.load(json_data_file)
        output = mp.Queue()
        processes = []
        for dataset in preprocess():
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
        return result_df, write_results_to_csv(result_df)


def summarize_results(results_df):
    summary_results = {}
    for i in range(len(CLASSIFIERS)):
        summary_results[results_df.columns[i].split('-')[1]] = {
            "baseline_accuracy": round(results_df.iloc[:, i].mean(), ACCURACY_FLOATING_POINT),
            "baseline_results": results_df.iloc[:, i],
            "experiments_accuracy": []
        }
    for i in range(len(CLASSIFIERS), len(results_df.columns)):
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
        results_string += 'T-Test: t = ' + str(round(t, ACCURACY_FLOATING_POINT)) + ', p = ' + \
                          str(round(p, ACCURACY_FLOATING_POINT)) + '\n'
        stat, p = stats.wilcoxon(baseline, experiment)
        results_string += 'Wilcoxon: s = ' + str(stat) + ', p = ' + str(round(p, ACCURACY_FLOATING_POINT)) + '\n'
        results_string += '\n'
    return results_string


def preprocess(normalization_method=DEFAULT_NORMALIZATION_METHOD):
    datasets = []
    for (name, dataset) in get_datasets():
        dataset.fillna(dataset.mean(), inplace=True)
        features = dataset.iloc[:, :-1]
        scaled_features = normalize(features.values, normalization_method)
        scaled_dataset = pd.DataFrame(scaled_features, index=features.index, columns=features.columns)
        datasets.append((name, pd.concat([scaled_dataset, dataset.iloc[:, -1]], axis=1)))
    return datasets


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
            print('Results file', input_file.split('/')[1], 'not found')
            df, input_file = get_experiments_results()
    else:
        df, input_file = get_experiments_results()
    stat_results = run_statistical_analysis(df)
    print(stat_results)
    config = configparser.RawConfigParser()
    config.read('ConfigFile.properties')
    if send_summary_email and SEND_EMAIL:
        send_email(config.get('EmailSection', 'email.user'), config.get('EmailSection', 'email.password'),
                   'ak091283@gmail.com', 'Finished Running', stat_results, input_file)
