import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import argparse
import os
import random
import torch
from torch import nn
from joblib import dump
from model import loadModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(prog = 'top', description='Train Classifier"')
      
parser.add_argument('-m', '--model', default='randomforest',
                   help='Choose one of the models ["logreg", "svc", "tree", "adaboost", "randomforest", "gradboost", "mlp", "fcnn"] (default: "randomforest")')

parser.add_argument('--C', type=float, default=100,
                    help='Initialize the "C" value (default: 100)')

parser.add_argument('--kernel', default='poly',
                    help='Choose "kernel" type (default: "poly")')

parser.add_argument('--max_depth', type=int, default=10,
                    help='Choose "max_depth" type (default: 10)')

parser.add_argument('--n_estimators', type=int, default=50,
                    help='Choose "max_depth" type (default: 50)')

parser.add_argument('--max_iter', type=int, default=1000,
                    help='Choose "max_iter" type (default: 1000)')

parser.add_argument('--hidden_layer_sizes', default='100,3',
                   help='Choose "hidden_layer_sizes" type (default: (100,3)')

parser.add_argument('--batch_size', type=int, default=32,
                    help='Initialize the "batch_size" value (default: 32)')

parser.add_argument('--test_size', type=float, default=0.2,
                    help='Initialize the "batch_size" value (default: 32)')

parser.add_argument('--epochs', type=int, default=10,
                    help='Initialize the "epochs" value (default: 10)')

parser.add_argument('--seed', type=int, default=42,
                    help='Set the seed value (default: 42)')

parser.add_argument('--save', type=bool, default=True,
                    help='Save model (default: True)')

parser.add_argument('--name', default='my_model',
                    help='Save model with "name" (default: "my_model")')

args = parser.parse_args()

if __name__ == "__main__":
    hls=tuple([int(i) for i in args.hidden_layer_sizes.split(',')])
    data = pd.read_csv('data/preprocessed_data.csv')
    X = data.drop(['Attrition'], axis=1).to_numpy()
    y = data['Attrition'].to_numpy()
    configs = dict(C=args.C, kernel=args.kernel, max_depth=args.max_depth,
                   n_estimators=args.n_estimators, max_iter=args.max_iter,
                   hidden_layer_sizes=hls, seed=args.seed)
    print('::::: Loading model :::::')
    model = loadModel(args.model, configs)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.seed)
    if args.model != 'fcnn':
        print('::::: Training :::::')
        model.fit(X_train, y_train)
        pred=model.predict(X_test)
    #else:
        #model = train(
    print('::::: Evaluation :::::')
    acc = accuracy_score(pred, y_test)
    f1 = f1_score(pred, y_test)
    p = precision_score(pred, y_test)
    r = recall_score(pred, y_test)
    print(f'::::: Model: {str(model)}')
    print(f'::::: Accuracy: {acc:.3f} ::::: F1 score: {f1:.3f}')
    print(f'::::: Precision: {p:.3f} ::::: Recall: {r:.3f}')
    if args.save:
        dump(model, 'models/' + args.model + '_' + args.name + '.joblib') 