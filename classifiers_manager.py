from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

CLASSIFIERS = [
    {
        "name": "decision_tree"
    },
    {
        "name": "nearest_neighbors"
    },
    # {
    #     "name": "random_forest"
    # },
    {
        "name": "ada_boost"
    },
    {
        "name": "gaussian_nb"
    }
]


def get_classifier(classifier_config):
    classifier_name = classifier_config['name']
    classifier_params = None
    if 'params' in classifier_config:
        classifier_params = classifier_config['params']
    if classifier_name == 'decision_tree':
        if classifier_params:
            return DecisionTreeClassifier(random_state=0, max_depth=classifier_params['max_depth'])
        return DecisionTreeClassifier(random_state=0)
    if classifier_name == 'nearest_neighbors':
        if classifier_params:
            return KNeighborsClassifier(n_neighbors=classifier_params['n_neighbors'])
        return KNeighborsClassifier()
    if classifier_name == 'random_forest':
        if classifier_params:
            return RandomForestClassifier(max_depth=classifier_params['max_depth'],
                                          n_estimators=classifier_params['n_estimators'], random_state=0)
        return RandomForestClassifier(n_estimators=10, random_state=0)
    if classifier_name == 'ada_boost':
        if classifier_params:
            return AdaBoostClassifier(n_estimators=classifier_params['n_estimators'], random_state=0)
        return AdaBoostClassifier(random_state=0)
    if classifier_name == 'gaussian_nb':
        return GaussianNB()
    raise NotImplementedError('Unsupported classifier')
