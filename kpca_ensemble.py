from os import listdir, makedirs
from os.path import isfile, join, exists
from kernels_manager import get_kernel
from classifiers_manager import get_classifier, CLASSIFIERS
from sklearn.model_selection import KFold, cross_val_score
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
DATAFRAME_COLUNMS = ['Dataset', 'Experiment', 'Classifier', 'Components', 'Accuracy']
DEFAULT_NUMBER_OF_KERNELS = 10


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


def write_results_to_csv(dataframe):
    if not exists(RESULTS_FOLDER):
        makedirs(RESULTS_FOLDER)
    current_time = strftime('%Y%m%d-%H%M%S', localtime())
    dataframe = dataframe[['Rank'] + DATAFRAME_COLUNMS]
    dataframe.to_csv(RESULTS_FOLDER + '/results-' + current_time + '.csv', index=False)


def write_results_to_json(dataset_name, data):
    if not exists(RESULTS_FOLDER):
        makedirs(RESULTS_FOLDER)
    current_time = strftime('%Y%m%d-%H%M%S', localtime())
    print(data)
    with open(RESULTS_FOLDER + '/' + current_time + '-' + dataset_name + '.json', 'w') as outfile:
        json.dump(data, outfile)


def get_total_number_of_experiments(experiments):
    count = 0
    for experiment_name, experiment_params in experiments.items():
        components = experiment_params['components'] if 'components' in experiment_params else [10, '0.5d']
        classifiers_list = experiment_params['classifiers'] if 'classifiers' in experiment_params \
            else CLASSIFIERS
        for _ in itertools.product(classifiers_list, components):
            count += 1
    return count


def build_results_json(result_list, kernels={}):
    return {
        "dataset": result_list[0],
        "experiment": result_list[1],
        "classifier": result_list[2],
        "components": result_list[3],
        "accuracy": result_list[4],
        "kernels": kernels
    }


def run_baseline(dataset_name, X, y):
    intermediate_results = []
    experiment_name = 'baseline'
    X = X.to_numpy()
    y = y.to_numpy()
    for classifier_config in CLASSIFIERS:
        accuracy = np.mean(cross_val_score(get_classifier(classifier_config), X, y, cv=DEFAULT_NUMBER_OF_KERNELS))
        result_list = [dataset_name, experiment_name, classifier_config['name'], 'N/A', round(accuracy, 5)]
        intermediate_results.append((result_list, build_results_json(result_list)))
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
        components = experiment_params['components'] if 'components' in experiment_params else [10, '0.5d']
        classifiers_list = experiment_params['classifiers'] if 'classifiers' in experiment_params \
            else CLASSIFIERS
        for experiment_config in itertools.product(classifiers_list, components):
            classifier_config = experiment_config[0]
            components_num = experiment_config[1]
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
            kf = KFold(n_splits=len(kernels))
            for i, (train_index, test_index) in enumerate(kf.split(X)):
                kernel, kernel_params = get_kernel(X, kernel_config, components_num)
                kernel_config = kernels[i]
                kernel_name = kernel_config['name']
                kernel_config['run_params'] = kernel_params
                X_train, X_test = kernel[train_index[0]: train_index[len(train_index) - 1], :], \
                    kernel[test_index[0]: test_index[len(test_index) - 1], :]
                y_train, y_test = y[train_index[0]: train_index[len(train_index) - 1]], \
                    y[test_index[0]: test_index[len(test_index) - 1]]
                clf = get_classifier(classifier_config)
                clf = clf.fit(X_train, y_train)
                results[kernel_name] = clf.predict(X_test)
            df = pd.DataFrame.from_dict(results)
            accuracy = np.round(metrics.accuracy_score(y_test, df.mode(axis=1).iloc[:, 0]), 5)
            result_list = [dataset_name, experiment_name, classifier_config['name'], components_num, accuracy]
            count += 1
            print(ctime(), '{0:.1%}'.format(float(count) / total_number_of_experiments), *result_list)
            intermediate_results.append((result_list, build_results_json(result_list, kernels)))
    print(ctime(), 'Finished running experiments on dataset', dataset_name)
    write_results_to_json(dataset_name, [intermediate_result[1] for intermediate_result in intermediate_results])
    output.put(intermediate_results)


def main():
    with open('experiments.json') as json_data_file:
        print(ctime(), 'Starting to run experiments')
        experiments = json.load(json_data_file)
        datasets = [f for f in listdir(DATASETS_FOLDER) if isfile(join(DATASETS_FOLDER, f))]
        datasets.sort()
        output = mp.Queue()
        df = pd.DataFrame([], columns=DATAFRAME_COLUNMS)
        processes = []
        for dataset in [(dataset, pd.read_csv(join(DATASETS_FOLDER, dataset), header=None)) for dataset in datasets]:
            p = mp.Process(target=run_experiments, args=(output, dataset, experiments))
            processes.append(p)
        for p in processes:
            p.start()
        results = [output.get() for _ in processes]
        print(ctime(), 'Finished running all experiments')
        for process_results in results:
            temp_df = pd.DataFrame([dataframe[0] for dataframe in process_results], columns=DATAFRAME_COLUNMS)
            temp_df = temp_df.sort_values(by='Accuracy', ascending=False)
            temp_df['Rank'] = range(1, 1 + len(temp_df))
            df = df.append(temp_df)
        write_results_to_csv(df)
        send_email('kagglemailsender', 'Amir!1@2#3$4', 'ak091283@gmail.com', 'Finished Running', df)


if __name__ == '__main__':
    main()
