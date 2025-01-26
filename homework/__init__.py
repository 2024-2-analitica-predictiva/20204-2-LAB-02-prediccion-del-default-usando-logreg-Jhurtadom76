import pandas as pd
test_data = pd.read_csv("files/input/test_data.csv.zip", index_col=False, compression="zip")
train_data = pd.read_csv("files/input/train_data.csv.zip", index_col=False, compression="zip")
test_data = test_data.rename(columns={'default payment next month': 'default'})
train_data = train_data.rename(columns={'default payment next month': 'default'})

test_data=test_data.drop(columns=['ID'])
train_data=train_data.drop(columns=['ID'])

import numpy as np


train_data = train_data.loc[train_data["MARRIAGE"] != 0]
train_data = train_data.loc[train_data["EDUCATION"] != 0]
test_data = test_data.loc[test_data["MARRIAGE"] != 0]
test_data = test_data.loc[test_data["EDUCATION"] != 0]


test_data['EDUCATION'] = test_data['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
train_data['EDUCATION'] = train_data['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
x_train=train_data.drop(columns="default")
y_train=train_data["default"]


x_test=test_data.drop(columns="default")
y_test=test_data["default"]

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import f_classif



categorical_features=["SEX","EDUCATION","MARRIAGE"]
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder="passthrough"
)


pipeline=Pipeline(
    [
        ("preprocessor",preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ]
)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score
param_grid = {
    'classifier__n_estimators': [100],  # Número de árboles
    'classifier__max_depth': [None],  # Profundidad máxima
    'classifier__min_samples_split': [10],  # Mínimo de muestras para dividir
    'classifier__min_samples_leaf': [4],   # Mínimo de muestras por hoja
    "classifier__max_features":[23]
}

model=GridSearchCV(
    pipeline,
    param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True
    )

model.fit(x_train, y_train)


import pickle
import os

models_dir = '../files/models'
os.makedirs(models_dir, exist_ok=True)

with open("../files/models/model.pkl","wb") as file:
    pickle.dump(model,file)


import json
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score

def calculate_and_save_metrics(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    metrics_train = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1_score': f1_score(y_train, y_train_pred, zero_division=0)
    }
    metrics_test = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_test_pred, zero_division=0)
    }
    output_dir = '../files/output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'metrics.json')
    with open(output_path, 'w') as f:  
        f.write(json.dumps(metrics_train) + '\n')
        f.write(json.dumps(metrics_test) + '\n')

from sklearn.metrics import confusion_matrix
def calculate_and_save_confusion_matrices(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    def format_confusion_matrix(cm, dataset_type):
        return {
            'type': 'cm_matrix',
            'dataset': dataset_type,
            'true_0': {
                'predicted_0': int(cm[0, 0]),
                'predicted_1': int(cm[0, 1])
            },
            'true_1': {
                'predicted_0': int(cm[1, 0]),
                'predicted_1': int(cm[1, 1])
            }
        }

    metrics = [
        format_confusion_matrix(cm_train, 'train'),
        format_confusion_matrix(cm_test, 'test')
    ]
    output_path = '../files/output/metrics.json'
    with open(output_path, 'a') as f:  
        for metric in metrics:
            f.write(json.dumps(metric) + '\n')

def main(model, X_train, X_test, y_train, y_test):
    import os
    os.makedirs('../files/output', exist_ok=True)
    calculate_and_save_metrics(model, X_train, X_test, y_train, y_test)
    calculate_and_save_confusion_matrices(model, X_train, X_test, y_train, y_test)
main(model, x_train, x_test, y_train, y_test)

from sklearn.metrics import confusion_matrix
def calculate_and_save_confusion_matrices(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    def format_confusion_matrix(cm, dataset_type):
        return {
            'type': 'cm_matrix',
            'dataset': dataset_type,
            'true_0': {
                'predicted_0': int(cm[0, 0]),
                'predicted_1': int(cm[0, 1])
            },
            'true_1': {
                'predicted_0': int(cm[1, 0]),
                'predicted_1': int(cm[1, 1])
            }
        }

    metrics = [
        format_confusion_matrix(cm_train, 'train'),
        format_confusion_matrix(cm_test, 'test')
    ]
    output_path = '../files/output/metrics.json'
    with open(output_path, 'a') as f:  
        for metric in metrics:
            f.write(json.dumps(metric) + '\n')
def main(model, X_train, X_test, y_train, y_test):
    import os
    os.makedirs('../files/output', exist_ok=True)
    calculate_and_save_metrics(model, X_train, X_test, y_train, y_test)
    calculate_and_save_confusion_matrices(model, X_train, X_test, y_train, y_test)
main(model, x_train, x_test, y_train, y_test)


