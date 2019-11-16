from os import listdir
from os.path import isfile, join
import json


class ExperimentsManager:
    def __init__(self):
        pass

    @staticmethod
    def get_experiments():
        with open('experiments.json') as json_data_file:
            return json.load(json_data_file)

    @staticmethod
    def get_datasets():
        datasets = [f for f in listdir('datasets') if isfile(join('datasets', f))]
        datasets.sort()
        return datasets
