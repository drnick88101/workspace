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

# set X4 to the top 4 features
X4 = X[['Glucose', 'Age', 'BMI', 'Systolic BP']]

# set X3 to the top 3 features
X3 = X[['Glucose', 'Age', 'BMI']]

# set X2 to the top 2 features
X2 = X[['Glucose', 'Age']]

# scores to evaluate the model
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score),
           'roc_auc_score' : make_scorer(roc_auc_score)}

# define k-fold
kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=42)

# define the model
# model=RandomForestClassifier(criterion='entropy', max_depth=7, max_features='log2', n_estimators=300)
# model=LogisticRegression(C=1, max_iter=100, penalty='l1', solver='saga')
# model=XGBClassifier(max_depth=2, learning_rate=0.01, gamma=0.6, subsample=0.1,verbosity=0)
# model=DecisionTreeClassifier(class_weight='balanced', criterion='log_loss', max_depth=8, max_features='sqrt', min_samples_leaf=5, splitter='best')
# model=SVC(C=1, gamma=1, kernel='linear')
# model=SVC(C=1, gamma=1, kernel='rbf')
# model=GaussianNB(var_smoothing=1.0)
# model=AdaBoostClassifier(algorithm='SAMME.R', learning_rate=0.001, n_estimators=50)
# model = KNeighborsClassifier(algorithm='auto', leaf_size=10, metric='manhattan', n_neighbors=3, p=1, weights='uniform')
# model = QuadraticDiscriminantAnalysis(reg_param=0.8, tol=0.0001)
model = MLPClassifier(activation='relu', alpha=0.05, hidden_layer_sizes=(50, 50, 50), learning_rate='constant', solver='adam')

# calculate the results
results = model_selection.cross_validate(model, X2, y, cv=kfold, scoring=scoring)

print("Accuracy: %.2f%% (%.2f%%)" % (results['test_accuracy'].mean()*100, results['test_accuracy'].std()*100))
print("Precision: %.2f%% (%.2f%%)" % (results['test_precision'].mean()*100, results['test_precision'].std()*100))
print("Recall: %.2f%% (%.2f%%)" % (results['test_recall'].mean()*100, results['test_recall'].std()*100))
print("F1 Score: %.2f%% (%.2f%%)" % (results['test_f1_score'].mean()*100, results['test_f1_score'].std()*100))
print("ROC AUC Score: %.2f%% (%.2f%%)" % (results['test_roc_auc_score'].mean()*100, results['test_roc_auc_score'].std()*100))
