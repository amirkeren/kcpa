import json


class ExperimentsManager:
    def __init__(self):
        with open('experiments.json') as json_data_file:
            self.experiments = json.load(json_data_file)

    def get_experiments(self):
        return self.experiments
