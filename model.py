from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

def loadModel(model, configs):
    if model=='logreg':
        return LogisticRegression(C=configs['C'], random_state=configs['seed'])
    elif model=='svc':
        return SVC(C=configs['C'], kernel=configs['kernel'], random_state=configs['seed'])
    elif model=='tree':
        return DecisionTreeClassifier(max_depth=configs['max_depth'], random_state=configs['seed'])
    elif model=='adaboost':
        return AdaBoostClassifier(n_estimators=configs['n_estimators'], random_state=configs['seed'])
    elif model=='randomforest':
        return RandomForestClassifier(n_estimators=configs['n_estimators'], random_state=configs['seed'])
    elif model=='gradboost':
        return GradientBoostingClassifier(n_estimators=configs['n_estimators'],
                                          max_depth=configs['max_depth'], random_state=configs['seed'])
    elif model=='mlp':
        return MLPClassifier(max_iter=configs['max_iter'],
                             hidden_layer_sizes=configs['hidden_layer_sizes'], random_state=configs['seed'])
    elif model=='fcnn':
        return FCNN(max_iter=configs['max_iter'],
                             hidden_layer_sizes=configs['hidden_layer_sizes'], random_state=configs['seed'])
    