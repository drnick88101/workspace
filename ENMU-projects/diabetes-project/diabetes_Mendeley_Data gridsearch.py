from tabnanny import verbose
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('~/workspace/datasets/diabetes_Mendeley_Data.csv')
df = data.copy()

# check for 0 values
print(df.isin([0]).sum())
print()

# check for null values
print(df.isnull().sum())

# drop the 'ID' column
df = df.drop('ID', axis = 1)

# drop the 'No_Pation' column
df = df.drop('No_Pation', axis = 1)

# replace 'Male' with 1 and 'Female' with 0
df['Gender'] = df['Gender'].replace('M', 1)
df['Gender'] = df['Gender'].replace('F', 0)
df['Gender'] = df['Gender'].replace('f', 0)

# replace 'N' with 0 and 'Y' with 1
df['CLASS'] = df['CLASS'].replace('N', 0)
df['CLASS'] = df['CLASS'].replace('N ', 0)
df['CLASS'] = df['CLASS'].replace('Y', 1)
df['CLASS'] = df['CLASS'].replace('Y ', 1)
df['CLASS'] = df['CLASS'].replace('P', 1)

# split the data into features and target
X = df.drop('CLASS', axis = 1)
y = df['CLASS']

# use standard scaler to scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# convert the data to a dataframe
X = pd.DataFrame(X, columns = df.columns[:-1])

# summarize class distribution
counter = Counter(y)
print(counter)

# over-sampling using SMOTE
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

# summarize class distribution
counter = Counter(y)
print(counter)

# create grid for Multi-Layer Perceptron GridSearchCV
grid = {
    'hidden_layer_sizes': [(10,30,10), (20,), (50,50,50), (50,100,50), (100,)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive', 'invscaling']
}

# run GridSearchCV model
model = GridSearchCV(estimator=MLPClassifier(), param_grid=grid, cv=5)

# fit the model
model.fit(X, y)

# print the best parameters
print(model.best_params_)
print()