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
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

data = pd.read_excel('~/workspace/datasets/Diabetes_Classification only.xlsx', sheet_name='Diabetes_Classification')
df = data.copy()

# drop the last 2 columns
df = df.drop(df.columns[[-2,-1]], axis=1)

# drop the first column
df = df.drop(df.columns[0], axis=1)

# drop the 'waist', 'hip', and 'Weight' columns
df = df.drop(['waist', 'hip', 'Weight'], axis=1)

# replace 'male' with 1 and 'female' with 0
df['Gender'] = df['Gender'].replace('male', 1)
df['Gender'] = df['Gender'].replace('female', 0)

# replace 'Diabetes' with 1 and 'No diabetes' with 0
df['Diabetes'] = df['Diabetes'].replace('Diabetes', 1)
df['Diabetes'] = df['Diabetes'].replace('No diabetes', 0)

# check for 0 values
print(df.isin([0]).sum())
print()

# check for null values
print(df.isnull().sum())

# split the data into features and target
X = df.drop('Diabetes', axis = 1)
y = df['Diabetes']

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