import json


class ExperimentsManager:
    def __init__(self):
        with open('experiments.json') as json_data_file:
            self.experiments_config = json.load(json_data_file)

    def print_config(self):
        return self.experiments_config
