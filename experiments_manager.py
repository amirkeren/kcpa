from os import listdir
from os.path import isfile, join
from kernels_manager import KernelsManager
from sklearn.tree import DecisionTreeClassifier
from kpca_utils import stepwise_kpca
from sklearn.model_selection import cross_val_score
import pandas as pd
import json


class ExperimentsManager:
    def __init__(self):
        with open('experiments.json') as json_data_file:
            self.experiments = json.load(json_data_file)
            datasets = [f for f in listdir('datasets') if isfile(join('datasets', f))]
            datasets.sort()
            self.datasets = [(dataset, pd.read_csv(join('datasets', dataset), header=None)) for dataset in datasets][1:]

    def get_next_experiment(self):
        return iter(self.experiments.values)

    def run(self, experiment):
        clf = DecisionTreeClassifier(random_state=42)
        for dataset, df in self.datasets:
            print(dataset)
            x = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            kernel_manager = KernelsManager(x)
            for kernel_config in experiment['kernels']:
                kernel = kernel_manager.get_kernel(kernel_config)
                X_pc = stepwise_kpca(kernel, experiment['components'])
            print(cross_val_score(clf, X_pc, df.iloc[:, -1], cv=2))