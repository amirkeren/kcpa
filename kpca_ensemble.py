from os import listdir, makedirs
from os.path import isfile, join, exists
from kernels_manager import get_kernel
from classifiers_manager import get_classifier, CLASSIFIERS
from sklearn.model_selection import KFold
from sklearn import metrics
from time import localtime, strftime, ctime
import pandas as pd
import multiprocessing as mp
import json
import itertools
import copy
import smtplib

DATASETS_FOLDER = 'datasets'
RESULTS_FOLDER = 'results'
DATAFRAME_COLUNMS = ['Dataset', 'Experiment', 'Classifier', 'Components', 'Accuracy']


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


def run_experiment(output, dataset, experiment, kernels, classifier_config, components_num):
    dataset_name = dataset[0]
    df = dataset[1]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    components_num = components_num if isinstance(components_num, int) else X.shape[1] // 2
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
        i += 1
    kf = KFold(n_splits=len(kernels), random_state=42)
    kf.get_n_splits(X)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        try:
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
        except:
            print('Error calculating kernel - ', dataset_name, kernel_name, kernel_params, classifier_config['name'],
                  components_num)
    df = pd.DataFrame.from_dict(results)
    accuracy = metrics.accuracy_score(y_test, df.mode(axis=1).iloc[:, 0])
    output.put(([dataset_name, experiment, classifier_config['name'], components_num, accuracy], {
        "experiment": experiment,
        "kernels": kernels,
        "components": components_num,
        "classifier": classifier_config,
        "dataset": dataset_name,
        "accuracy": accuracy
    }))


def write_results_to_csv(dataframe):
    if not exists(RESULTS_FOLDER):
        makedirs(RESULTS_FOLDER)
    current_time = strftime('%Y%m%d-%H%M%S', localtime())
    dataframe.to_csv(RESULTS_FOLDER + '/results-' + current_time + '.csv')


def write_results_to_json(dataset_name, data):
    if not exists(RESULTS_FOLDER):
        makedirs(RESULTS_FOLDER)
    current_time = strftime('%Y%m%d-%H%M%S', localtime())
    filename = RESULTS_FOLDER + '/' + current_time + '-' + dataset_name + '.json'
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)
    return json.dumps(data)


def main():
    with open('experiments.json') as json_data_file:
        print(ctime(), 'Starting to run experiments')
        experiments = json.load(json_data_file)
        datasets = [f for f in listdir(DATASETS_FOLDER) if isfile(join(DATASETS_FOLDER, f))]
        datasets.sort()
        output = mp.Queue()
        df = pd.DataFrame([], columns=DATAFRAME_COLUNMS)
        for dataset in [(dataset, pd.read_csv(join(DATASETS_FOLDER, dataset), header=None)) for dataset in datasets]:
            dataset_name = dataset[0]
            print(ctime(), 'Starting to run experiments on dataset', dataset_name)
            processes = []
            for experiment_name, experiment_params in experiments.items():
                components = experiment_params['components'] if 'components' in experiment_params else [10, '0.5d']
                classifiers_list = experiment_params['classifiers'] if 'classifiers' in experiment_params \
                    else CLASSIFIERS
                for experiment_config in itertools.product(classifiers_list, components):
                    classifiers = experiment_config[0]
                    n_components = experiment_config[1]
                    p = mp.Process(target=run_experiment,
                                   args=(output, dataset, experiment_name, experiment_params['kernels'], classifiers,
                                         n_components))
                    processes.append(p)
            for p in processes:
                p.start()
            results = [output.get() for p in processes]
            df = df.append(pd.DataFrame([dataframe[0] for dataframe in results], columns=DATAFRAME_COLUNMS))
            print(ctime(), 'Finished running experiments on dataset', dataset_name)
            write_results_to_json(dataset_name, [dataframe[1] for dataframe in results])
        print(ctime(), 'Finished running all experiments')
        result_df = df.sort_values(['Dataset', 'Accuracy'], ascending=[True, False])
        write_results_to_csv(result_df)
        send_email('kagglemailsender', 'Amir!1@2#3$4', 'ak091283@gmail.com', 'Finished Running', result_df)


if __name__ == '__main__':
    main()
