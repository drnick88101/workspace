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

data = pd.read_csv('~/workspace/datasets/diabetes_kaggle_9_cols.csv')
df = data.copy()

# replace 0's with NaN
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

# replace NaN's with the mean of the column
df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)
df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace = True)
df['Insulin'].fillna(df['Insulin'].mean(), inplace = True)
df['BMI'].fillna(df['BMI'].mean(), inplace = True)

# split the data into X and y
X = df.drop('Outcome', axis = 1)
y = df['Outcome']

# use standard scaler to scale the data
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

# set X4 to the top 4 features
X4 = X[['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction']]

# set X3 to the top 3 features
X3 = X[['Glucose', 'BMI', 'Age']]

# set X2 to the top 2 features
X2 = X[['Glucose', 'BMI']]

# summarize the new class distribution
counter = Counter(y)
print(counter)

# scores to evaluate the model
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score),
           'roc_auc_score' : make_scorer(roc_auc_score)}

# define k-fold
kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=42)

# define the model
# model=RandomForestClassifier(criterion='entropy', max_depth=8, max_features='log2', n_estimators=100)
# model=LogisticRegression(C=1, max_iter=100, penalty='l1', solver='liblinear')
# model=XGBClassifier(max_depth=10, learning_rate=0.2, gamma=0, subsample=0.7,verbosity=0)
# model=DecisionTreeClassifier(class_weight='balanced', criterion='log_loss', max_depth=6, max_features='log2', min_samples_leaf=5, splitter='best')
# model=SVC(C=0.1, gamma=1, kernel='linear')
# model=SVC(C=10, gamma=1, kernel='rbf')
# model=GaussianNB(var_smoothing=0.0015199110829529332)
# model=AdaBoostClassifier(algorithm='SAMME.R', learning_rate=0.1, n_estimators=300)
# model = KNeighborsClassifier(algorithm='auto', leaf_size=10, metric='euclidean', n_neighbors=15, weights='distance')
# model = QuadraticDiscriminantAnalysis(reg_param=0.3, tol=0.0001)
model = MLPClassifier(activation='relu', alpha=0.05, hidden_layer_sizes=(50, 50, 50), learning_rate='invscaling', solver='adam')

# calculate the results
results = model_selection.cross_validate(model, X3, y, cv=kfold, scoring=scoring)

print("Accuracy: %.2f%% (%.2f%%)" % (results['test_accuracy'].mean()*100, results['test_accuracy'].std()*100))
print("Precision: %.2f%% (%.2f%%)" % (results['test_precision'].mean()*100, results['test_precision'].std()*100))
print("Recall: %.2f%% (%.2f%%)" % (results['test_recall'].mean()*100, results['test_recall'].std()*100))
print("F1 Score: %.2f%% (%.2f%%)" % (results['test_f1_score'].mean()*100, results['test_f1_score'].std()*100))
print("ROC AUC Score: %.2f%% (%.2f%%)" % (results['test_roc_auc_score'].mean()*100, results['test_roc_auc_score'].std()*100))


