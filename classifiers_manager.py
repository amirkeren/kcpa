from distutils import util
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from rotation_forest import RotationForestClassifier

CLASSIFIERS = [
    {
        "name": "decision_tree",
        "ensemble": "False"
    },
    {
        "name": "random_forest",
        "ensemble": "True"
    },
    {
        "name": "gradient_boosting",
        "ensemble": "True"
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
    if classifier_name == 'decision_stump':
        return DecisionTreeClassifier(random_state=0, max_depth=2)
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
    if classifier_name == 'rotation_forest':
        if classifier_params:
            return RotationForestClassifier(max_depth=classifier_params['max_depth'],
                                            n_estimators=classifier_params['n_estimators'], random_state=0)
        return RotationForestClassifier(random_state=0)
    if classifier_name == 'bagging':
        if classifier_params:
            return BaggingClassifier(n_estimators=classifier_params['n_estimators'], random_state=0)
        return BaggingClassifier(random_state=0)
    if classifier_name == 'gradient_boosting':
        if classifier_params:
            return GradientBoostingClassifier(max_depth=classifier_params['max_depth'],
                                              n_estimators=classifier_params['n_estimators'], random_state=0)
        return GradientBoostingClassifier(random_state=0)
    raise NotImplementedError('Unsupported classifier')


def is_ensemble_classifier(classifier_name):
    for classifier in CLASSIFIERS:
        if classifier['name'] == classifier_name:
            return bool(util.strtobool(classifier['ensemble']))
    return False
