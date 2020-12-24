import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import argparse
import os
import random
import torch
from torch import nn
from joblib import dump, load 
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
                   help='Choose one of the models ["svc", "randomforest", "gradboost", "mlp", "fcnn"] or your own. (default: "randomforest")')

parser.add_argument('--test_size', type=float, default=0.2,
                    help='Initialize the "batch_size" value (default: 32)')

parser.add_argument('--batch_size', type=int, default=32,
                    help='Initialize the "batch_size" value (default: 32)')

parser.add_argument('--seed', type=int, default=42,
                    help='Set the seed value (default: 42)')

parser.add_argument('--average', type=bool, default=False,
                    help='Find average performance (default: False)')

parser.add_argument('--full', type=bool, default=False,
                    help='Find average performance (default: False)')

parser.add_argument('--seed_list', default='13,42,666',
                    help='Set seed values (default: 42)')

args = parser.parse_args()

if __name__ == "__main__":
    seeds = [int(i) for i in args.seed_list.split(',')]
    data = pd.read_csv('data/preprocessed_data.csv')
    X = data.drop(['Attrition'], axis=1).to_numpy()
    y = data['Attrition'].to_numpy()
   
    model=load('models/' + args.model + '.joblib')
    acc=[]
    f1=[]
    p=[]
    r=[]
    for seed in seeds:
        set_seed(seed)
        if not args.full:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=seed)
        else:
            X_test = X
            y_test = y
        pred = model.predict(X_test)
        
        acc.append(accuracy_score(pred, y_test))
        f1.append(f1_score(pred, y_test))
        p.append(precision_score(pred, y_test))
        r.append(recall_score(pred, y_test))
    print(f'::::: Model: {str(model)}')
    print(f'::::: Accuracy: {np.mean(acc):.3f} +/- {np.std(acc):.3f} ::::: F1 score: {np.mean(f1):.3f} +/- {np.std(f1):.3f}')
    print(f'::::: Precision: {np.mean(p):.3f} +/- {np.std(p):.3f} ::::: Recall: {np.mean(r):.3f} +/- {np.std(r):.3f}')