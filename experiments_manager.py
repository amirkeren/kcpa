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

    def get_experiments(self):
        return self.experiments.items()

    def run(self, experiment):
        clf = DecisionTreeClassifier(random_state=42)
        # parallelize it
        for dataset, df in self.datasets:
            print(dataset)
            x = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            kernel_manager = KernelsManager(x)
            kernel_ensamble = []
            for kernel_config in experiment['kernels']:
                x_pc = stepwise_kpca(kernel_manager.get_kernel(kernel_config), experiment['components'])
                kernel_ensamble.append(x_pc)
            for kernel in kernel_ensamble:
                print(cross_val_score(clf, kernel, y, cv=2))
