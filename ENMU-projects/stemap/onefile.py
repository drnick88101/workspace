import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
import sklearn.preprocessing as preprocessing
from pyod.utils.data import evaluate_print
from sklearn.decomposition import PCA
from pyod.models.knn import KNN
from pyod.models.ocsvm import OCSVM
from pyod.models.abod import ABOD
from pyod.models.lof import LOF
from pyod.models.iforest import IForest

# for loop to evaluate the model 10 times
for i in range(1, 16):
    roco = 0.0 
    rock = 0.0 
    roca  = 0.0
    rocl = 0.0
    roci = 0.0
    data = pd.read_csv('week 3/data/data' + str(i) + '.csv', header=0)
    print('Filename: data' + str(i) + '.csv')

    # for loop to evaluate the model 10 times
    for j in range(10):
        # remove the log columns
        for col in data.columns:
            if 'log' in col:
                data.drop(col, axis=1, inplace=True)

        # remove any infinite values
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        # group the 'marker' column by 'Natural'
        data_grouped = data.sort_values('marker', ascending=False)

        # split the data into training and testing sets
        # training data is half of the 'natural' data
        # testing data is the other half of the 'natural' data pluse the 'attack' data
        num = data['marker'].value_counts()['Natural']//2
        x_train = data.groupby('marker').get_group('Natural').sample(frac=0.5)
        data_natural = data.groupby('marker').get_group('Natural').drop(x_train.index)
        data_attack = data.groupby('marker').get_group('Attack').sample(n=num)
        x_test = pd.concat([data_natural, data_attack])
        y_test = x_test['marker']

        # drop the marker columns
        x_train.drop('marker', axis=1, inplace=True)
        x_test.drop('marker', axis=1, inplace=True)

        # normalize the data with min-max scaler
        # the same scaler is used for both training and testing data
        scaler = preprocessing.MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # PCA
        pca = PCA(n_components=2)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)

        # replace 'Attack' with 1 and 'Natural' with 0
        y_test.replace('Attack', 1, inplace=True)
        y_test.replace('Natural', 0, inplace=True)

        # train OCSVM
        clf_name = 'OneClassSVM'
        clf = OCSVM()
        clf.fit(x_train)

        # get the testing data anomaly score
        y_score = clf.decision_function(x_test)

        # evaluate the results and print them
        roco += np.round(roc_auc_score(y_test, y_score), decimals=4)
        # evaluate_print(clf_name, y_test, y_score)

        # train ABOD detector
        clf_name = 'ABOD'
        clf = ABOD()
        clf.fit(x_train)

        # get the testing data anomaly score
        y_score = clf.decision_function(x_test)

        # evaluate the results and print them
        roca += np.round(roc_auc_score(y_test, y_score), decimals=4)
        # evaluate_print(clf_name, y_test, y_score)

        # train KNN detector
        clf_name = 'KNN'
        clf = KNN()
        clf.fit(x_train)

        # get the testing data anomaly score
        y_score = clf.decision_function(x_test)

        # evaluate the results and print them
        rock += np.round(roc_auc_score(y_test, y_score), decimals=4)
        # evaluate_print(clf_name, y_test, y_score)

        # train LOF detector
        clf_name = 'LOF'
        clf = LOF()
        clf.fit(x_train)

        # get the testing data anomaly score
        y_score = clf.decision_function(x_test)

        # evaluate the results and print them
        rocl += np.round(roc_auc_score(y_test, y_score), decimals=4)
        # evaluate_print(clf_name, y_test, y_score)

        # train IForest detector
        clf_name = 'IForest'
        clf = IForest()
        clf.fit(x_train)

        # get the testing data anomaly score
        y_score = clf.decision_function(x_test)

        # evaluate the results and print them
        roci += np.round(roc_auc_score(y_test, y_score), decimals=4)
        # evaluate_print(clf_name, y_test, y_score)


    print('Average ROC AUC for OneClassSVM:' + str(np.round(roco/10, decimals=4)))
    print('Average ROC AUC for ABOD:' + str(np.round(roca/10, decimals=4)))
    print('Average ROC AUC for KNN:' + str(np.round(rock/10, decimals=4)))
    print('Average ROC AUC for LOF:' + str(np.round(rocl/10, decimals=4)))
    print('Average ROC AUC for IForest:' + str(np.round(roci/10, decimals=4)))
    print()