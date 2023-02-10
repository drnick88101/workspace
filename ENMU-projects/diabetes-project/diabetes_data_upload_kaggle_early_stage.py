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

data = pd.read_csv('~/workspace/datasets/diabetes_data_upload_kaggle_early_stage.csv')
df = data.copy()

# replace 'Male' with 1 and 'Female' with 0
df['Gender'] = df['Gender'].replace('Male', 1)
df['Gender'] = df['Gender'].replace('Female', 0)

# replace 'Yes' with 1 and 'No' with 0
df[['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity']] = df[['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity']].replace('Yes', 1)
df[['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity']] = df[['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity']].replace('No', 0)

# replace 'Positive' with 1 and 'Negative' with 0
df['class'] = df['class'].replace('Positive', 1)
df['class'] = df['class'].replace('Negative', 0)

# split the data into X and y
X = df.drop('class', axis = 1)
y = df['class']

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
X4 = X[['Age', 'Gender', 'Polyuria', 'Polydipsia']]

# set X3 to the top 3 features
X3 = X[['Age', 'Polyuria', 'Polydipsia']]

# set X2 to the top 2 features
X2 = X[['Polyuria', 'Polydipsia']]

# scores to evaluate the model
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score),
           'roc_auc_score' : make_scorer(roc_auc_score)}

# define k-fold
kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=42)

# define the model
#model=RandomForestClassifier(criterion='gini', max_depth=8, max_features='log2', n_estimators=100)
#model=LogisticRegression(C=10, max_iter=100, penalty='l2', solver='liblinear')
#model=XGBClassifier(max_depth=6, learning_rate=0.3, gamma=0.1, subsample=0.7)
#model=DecisionTreeClassifier(class_weight='balanced', criterion='log_loss', max_depth=10, max_features='auto', min_samples_leaf=5, splitter='best')
#model=SVC(C=1, gamma=1, kernel='linear')
#model=SVC(C=10, gamma=0.1, kernel='rbf')
#model=GaussianNB(var_smoothing=0.0001)
#model=AdaBoostClassifier(algorithm='SAMME.R', learning_rate=1, n_estimators=450)
model = KNeighborsClassifier(algorithm='auto', leaf_size=10, metric='euclidean', n_neighbors=3, p=1, weights='distance')
#model = QuadraticDiscriminantAnalysis(reg_param=0.1, tol=0.0001)
#model = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=(50, 50, 50), learning_rate='adaptive', solver='adam')

# calculate the results
results = model_selection.cross_validate(model, X2, y, cv=kfold, scoring=scoring)

print("Accuracy: %.2f%% (%.2f%%)" % (results['test_accuracy'].mean()*100, results['test_accuracy'].std()*100))
print("Precision: %.2f%% (%.2f%%)" % (results['test_precision'].mean()*100, results['test_precision'].std()*100))
print("Recall: %.2f%% (%.2f%%)" % (results['test_recall'].mean()*100, results['test_recall'].std()*100))
print("F1 Score: %.2f%% (%.2f%%)" % (results['test_f1_score'].mean()*100, results['test_f1_score'].std()*100))
print("ROC AUC Score: %.2f%% (%.2f%%)" % (results['test_roc_auc_score'].mean()*100, results['test_roc_auc_score'].std()*100))