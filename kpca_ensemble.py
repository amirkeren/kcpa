from distutils import util
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from os import listdir, makedirs, remove
from os.path import isfile, join, exists
from kernel import Kernel
from classifiers_manager import get_classifier, CLASSIFIERS, is_ensemble_classifier
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from normalization import normalize, Normalization
from sklearn import metrics
from time import localtime, strftime, ctime
from scipy import stats
from pathlib import Path
import operator
import datetime
import pandas as pd
import numpy as np
import multiprocessing as mp
import sys
import json
import itertools
import smtplib
import configparser
import random

RUN_PARALLEL = True
RUN_ON_LARGE_DATASETS = True
SEND_EMAIL = True
PRINT_TO_STDOUT = False
PROVIDE_SEED = False
REMOVE_INVALID_RESULTS = True
CAP_DATASETS_AT = -1
LOGFILE_NAME = 'logs/output-' + strftime("%d%m%Y-%H%M") + '.log'
DATASETS_FOLDER = 'datasets'
LARGE_DATASETS_FOLDER = 'large_datasets'
RESULTS_FOLDER = 'results'
ACCURACY_FLOATING_POINT = 5
DEFAULT_NUMBER_OF_FOLDS = 10  # 2
DEFAULT_NORMALIZATION_METHOD_PREPROCESS = Normalization.STANDARD
DEFAULT_NORMALIZATION_METHOD_PRECOMBINE = Normalization.STANDARD
# grid searchable
DEFAULT_NUMBER_OF_MEMBERS = [11]
DEFAULT_NUMBER_OF_COMPONENTS = ['10']


def print_info(message, print_to_stdout=PRINT_TO_STDOUT):
    if print_to_stdout:
        print(ctime(), message)
    log_file.write(ctime() + ' ' + message + '\n')


def send_email(user, pwd, recipient, subject, body, file):
    print_info('Sending summary email')
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
    print_info('Summary email sent')


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
    filename = RESULTS_FOLDER + '/results-' + current_time + '.csv'
    if REMOVE_INVALID_RESULTS:
        dataframe = dataframe.drop(columns=dataframe.columns[(dataframe == -100).any()])
    averages = dataframe.mean(axis=0)
    best_baseline_accuracy = best_experiment_accuracy = -1
    for index, value in averages.items():
        if index.startswith('baseline'):
            if value > best_baseline_accuracy:
                best_baseline_accuracy = value
                best_baseline = index
        else:
            if value > best_experiment_accuracy:
                best_experiment_accuracy = value
                best_experiment = index
    cols = list(dataframe.columns.values)
    cols.pop(cols.index(best_baseline))
    cols.pop(cols.index(best_experiment))
    dataframe = dataframe[[best_experiment, best_baseline] + cols]
    dataframe.to_csv(filename)
    return filename


def get_experiment_parameters(experiment_params):
    classifiers_list = experiment_params['classifiers'] if 'classifiers' in experiment_params \
        else CLASSIFIERS
    components = experiment_params['components'] if 'components' in experiment_params \
        else DEFAULT_NUMBER_OF_COMPONENTS
    ensemble_size = experiment_params['ensemble_size'] if 'ensemble_size' in experiment_params \
        else DEFAULT_NUMBER_OF_MEMBERS
    return classifiers_list, components, ensemble_size


def get_total_number_of_experiments(experiments):
    count = 0
    for _, experiment_params in experiments.items():
        for classifier_config in itertools.product(*get_experiment_parameters(experiment_params)):
            if not bool(util.strtobool(classifier_config[0]['ensemble'])):
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


def run_experiments(dataset):
    with open('experiments.json') as json_data_file:
        experiments = json.load(json_data_file)
    dataset_name = dataset[0].split('\\')[1]
    global log_file
    log_file = open(LOGFILE_NAME, 'a')
    print_info('Starting to run experiments on dataset ' + dataset_name)
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
        print_info('Starting to run experiments ' + experiment_name + ' on ' + dataset_name)
        for experiment_config in itertools.product(*get_experiment_parameters(experiment_params)):
            try:
                classifier_config = experiment_config[0]
                if bool(util.strtobool(classifier_config['ensemble'])):
                    continue
                components_str = experiment_config[1]
                members_num = experiment_config[2]
                components_num = components_str if isinstance(components_str, int) else \
                    round(X.shape[1] * float(components_str[:-1]))
                if PROVIDE_SEED:
                    random.seed(30)
                splits, splits_copy = itertools.tee(splits)
                accuracies = []
                i = 0
                for train_index, test_index in splits_copy:
                    results = {}
                    X_train, X_test = X.values[train_index], X.values[test_index]
                    y_train, y_test = y.values[train_index], y.values[test_index]
                    kernels_and_classifiers = []
                    for _ in range(members_num):
                        rbf_kernel = Kernel({'name': 'rbf'}, components_num, avg_euclid_distances,
                                            max_euclid_distances,
                                            normalization_method=DEFAULT_NORMALIZATION_METHOD_PRECOMBINE,
                                            random=random)
                        poly_kernel = Kernel({'name': 'poly'}, components_num, avg_euclid_distances,
                                             max_euclid_distances,
                                             normalization_method=DEFAULT_NORMALIZATION_METHOD_PRECOMBINE,
                                             random=random)
                        sigmoid_kernel = Kernel({'name': 'sigmoid'}, components_num, avg_euclid_distances,
                                                max_euclid_distances,
                                                normalization_method=DEFAULT_NORMALIZATION_METHOD_PRECOMBINE,
                                                random=random)
                        kernels = [rbf_kernel, poly_kernel, sigmoid_kernel]
                        sub_splits = RepeatedStratifiedKFold(n_splits=DEFAULT_NUMBER_OF_FOLDS,
                                                             n_repeats=5 if DEFAULT_NUMBER_OF_FOLDS == 2 else 1,
                                                             random_state=0).split(X_train, y_train)
                        temp_accuracies = {}
                        for i, kernel in enumerate(kernels):
                            kernel_accuracies = []
                            sub_splits, sub_splits_copy = itertools.tee(sub_splits)
                            for sub_train_index, sub_test_index in sub_splits_copy:
                                try:
                                    X_sub_train, X_sub_test = X_train[sub_train_index], X_train[sub_test_index]
                                    y_sub_train, y_sub_test = y_train[sub_train_index], y_train[sub_test_index]
                                    embedded_train = kernel.calculate_kernel(X_sub_train)
                                    embedded_test = kernel.calculate_kernel(X_sub_test, is_test=True)
                                    clf = get_classifier(classifier_config)
                                    clf.fit(embedded_train, y_sub_train)
                                    predictions = clf.predict(embedded_test)
                                    kernel_accuracies.append(metrics.accuracy_score(y_sub_test, predictions))
                                except:
                                    kernel_accuracies.append(-1)
                                    print_info(dataset_name + ' - Failed to calculate accuracy for kernel ' +
                                               kernel.kernel_name)
                            temp_accuracies[i] = round(np.asarray(kernel_accuracies).mean(), ACCURACY_FLOATING_POINT)
                        member = kernels[max(temp_accuracies.items(), key=operator.itemgetter(1))[0]]
                        embedded_train = member.calculate_kernel(X_train)
                        clf = get_classifier(classifier_config)
                        clf.fit(embedded_train, y_train)
                        kernels_and_classifiers.append((member, clf))
                    for kernel, clf in kernels_and_classifiers:
                        embedded_test = kernel.calculate_kernel(X_test, is_test=True)
                        predictions = clf.predict(embedded_test)
                        results[(kernel, clf)] = predictions
                    results_df = pd.DataFrame.from_dict(results)
                    ensemble_vote = results_df.mode(axis=1).iloc[:, 0]
                    accuracies.append(metrics.accuracy_score(y_test, ensemble_vote))
                    i += 1
                accuracy = round(np.asarray(accuracies).mean(), ACCURACY_FLOATING_POINT)
                intermediate_results.setdefault(dataset_name, []).append(
                    (build_experiment_key(experiment_name, classifier_config['name'], components_str,
                                          DEFAULT_NUMBER_OF_FOLDS, members_num),
                     accuracy))
                count += 1
                if PRINT_TO_STDOUT:
                    str_to_print = build_experiment_key(experiment_name, classifier_config['name'], components_str,
                                                        DEFAULT_NUMBER_OF_FOLDS, members_num)
                    print_info('{0:.1%}'.format(float(count) / total_number_of_experiments) + ' ' + dataset_name +
                        ' ' + str_to_print)
                else:
                    kernels = [key[0] for key in kernels_and_classifiers]
                    str_to_print = build_experiment_key(experiment_name, classifier_config['name'], components_str,
                                                        DEFAULT_NUMBER_OF_FOLDS, members_num, kernels)
                    print_info('{0:.1%}'.format(float(count) / total_number_of_experiments) + ' ' + dataset_name +
                               ' ' + str_to_print)
            except Exception as e:
                print_info('Failed to run experiment ' + experiment_name + ' on dataset ' + dataset_name +
                           ' with exception ' + str(e))
                count += 1
                intermediate_results.setdefault(dataset_name, []).append(
                    (build_experiment_key(experiment_name, classifier_config['name'], components_str,
                                          DEFAULT_NUMBER_OF_FOLDS, members_num), -100))
        print_info('Finished running experiment ' + experiment_name + ' on dataset ' + dataset_name)
    print_info('Finished running experiments on dataset ' + dataset_name)
    log_file.close()
    return intermediate_results


def get_datasets():
    datasets = []
    if RUN_ON_LARGE_DATASETS:
        datasets.extend([LARGE_DATASETS_FOLDER + '\\' + f for f in listdir(LARGE_DATASETS_FOLDER)
                         if isfile(join(LARGE_DATASETS_FOLDER, f))])
    datasets.extend([DATASETS_FOLDER + '\\' + f for f in listdir(DATASETS_FOLDER) if
                     isfile(join(DATASETS_FOLDER, f))])
    return [(dataset, pd.read_csv(dataset, header=None)) for dataset in datasets]


def get_experiments_results():
    pool = mp.Pool(mp.cpu_count()) if RUN_PARALLEL else mp.Pool(1)
    results = pool.map(run_experiments, [dataset for dataset in preprocess()])
    pool.close()
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
        if is_ensemble_classifier(results_df.columns[i].split("-")[1]):
            continue
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
    overall_best_experiment_accuracy = -1
    overall_best_experiment = {}
    for key, value in summary_results.items():
        if value['best_experiment']['accuracy'] > overall_best_experiment_accuracy:
            overall_best_experiment_accuracy = value['best_experiment']['accuracy']
            overall_best_experiment = value['best_experiment']
    for key in list(summary_results):
        if not is_ensemble_classifier(key):
            del summary_results[key]
    summary_results['best_experiment'] = overall_best_experiment
    return summary_results


def run_statistical_analysis(results_df):
    print_info('Run statistical analysis on results')
    summarized_results = summarize_results(results_df)
    best_experiment = summarized_results['best_experiment']
    results_string = 'Best experiment is - ' + str(best_experiment['experiment']) + '\n'
    for key, value in summarized_results.items():
        if key == 'best_experiment':
            continue
        results_string += '\nBest experiment vs. ' + key + '\n'
        results_string += compare_experiments(best_experiment, value, key) + '\n'
    return results_string


def compare_experiments(experiment, baseline, baseline_name):
    results_string = ''
    if experiment['accuracy'] >= baseline['baseline_accuracy']:
        baseline_results = baseline['baseline_results']
        experiment_results = experiment['experiment_results']
        results_string += 'Experiment wins: ' + str(experiment['accuracy']) + ' >= ' + \
                          str(baseline['baseline_accuracy']) + '\n'
        try:
            t, p = stats.ttest_ind(baseline_results, experiment_results)
            results_string += 'T-Test: t = ' + str(round(t, ACCURACY_FLOATING_POINT)) + ', p = ' + \
                              str(round(p, ACCURACY_FLOATING_POINT)) + '\n'
        except Exception as e:
            results_string += 'Failed to run t-test - ' + str(e) + '\n'
        try:
            stat, p = stats.wilcoxon(baseline_results, experiment_results)
            results_string += 'Wilcoxon: s = ' + str(stat) + ', p = ' + str(round(p, ACCURACY_FLOATING_POINT)) + '\n'
        except Exception as e:
            results_string += 'Failed to run Wilcoxon test - ' + str(e) + '\n'
    else:
        results_string += baseline_name + ' wins: ' + str(baseline['baseline_accuracy']) + ' > ' + \
                          str(experiment['accuracy']) + '\n'
    return results_string


def preprocess(normalization_method=DEFAULT_NORMALIZATION_METHOD_PREPROCESS, cap=CAP_DATASETS_AT):
    datasets = []
    for (name, dataset) in get_datasets():
        if 0 < cap < len(dataset.index):
            dataset = dataset.sample(cap, random_state=0)
        dataset.fillna(dataset.mean(), inplace=True)
        features = dataset.iloc[:, :-1]
        scaled_features = normalize(features.values, normalization_method)
        scaled_dataset = pd.DataFrame(scaled_features, index=features.index, columns=features.columns)
        datasets.append((name, pd.concat([scaled_dataset, dataset.iloc[:, -1]], axis=1)))
    return datasets


if __name__ == '__main__':
    if isfile(LOGFILE_NAME):
        remove(LOGFILE_NAME)
    global log_file
    Path('logs').mkdir(parents=True, exist_ok=True)
    log_file = open(LOGFILE_NAME, 'a')
    input_file = None
    df = None
    send_summary_email = True
    for arg in sys.argv[1:]:
        input_file = arg
    start = datetime.datetime.now()
    if input_file:
        send_summary_email = False
        input_file = 'results/' + input_file
        if isfile(input_file):
            print_info('Results file found')
            df = pd.read_csv(input_file, index_col=0)
        else:
            print_info('Results file ' + input_file.split('/')[1] + ' not found')
            print_info('Starting to run experiments')
            log_file.close()
            df, input_file = get_experiments_results()
    else:
        print_info('Starting to run experiments')
        log_file.close()
        df, input_file = get_experiments_results()
    log_file = open(LOGFILE_NAME, 'a')
    difference = datetime.datetime.now() - start
    print_info('Finished running all experiments')
    print_info('Total run time is ' + str(difference))
    stat_results = run_statistical_analysis(df)
    print_info(stat_results, print_to_stdout=True)
    config = configparser.RawConfigParser()
    config.read('ConfigFile.properties')
    if send_summary_email and SEND_EMAIL:
        send_email(config.get('EmailSection', 'email.user'), config.get('EmailSection', 'email.password'),
                   'ak091283@gmail.com', 'Finished Running', 'Total run time is ' + str(difference) + '\n\n' +
                   stat_results, input_file)
    log_file.close()
