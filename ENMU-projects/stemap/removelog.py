import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.preprocessing as preprocessing
import pyod.models.knn as knn
from pyod.utils.data import evaluate_print

data = pd.read_csv('week 3/data/data1.csv', header=0)

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

# replace 'Attack' with 1 and 'Natural' with 0
y_test.replace('Attack', 1, inplace=True)
y_test.replace('Natural', 0, inplace=True)

# train kNN detector
knn_name = 'KNN'
knn_detector = knn.KNN()
knn_detector.fit(x_train)

# get the testing data anomaly score
y_score = knn_detector.decision_function(x_test)

# evaluate the results
evaluate_print(knn_name, y_test, y_score)