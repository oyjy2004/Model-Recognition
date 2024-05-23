import numpy as np
import pandas as pd
from PLA import PLA
import math

data = pd.read_csv("F:\Codes and Works\python_work\Model_Recognition_and_Machine_Learning\实验数据集\Iris数据集\iris.csv")
data_list = data.values.tolist()
data_arr = np.array(data_list)
X = data_arr[:, 1:5]
X = X.astype(np.double)
y = data_arr[:, 5]
# print(data_arr)
# print(X)
# print(y)
X_1 = X[:50]
y_1 = y[:50]
X_2 = X[50:100]
y_2 = y[50:100]
X_3 = X[100:150]
y_3 = y[100:150]

indices = np.arange(X_1.shape[0])
np.random.shuffle(indices)
X_1 = X_1[indices]
y_1 = y_1[indices]
X_2 = X_2[indices]
y_2 = y_2[indices]
X_3 = X_3[indices]
y_3 = y_3[indices]

X_train = np.vstack((X_1[:30], X_2[:30], X_3[:30]))
y_train = np.hstack((y_1[:30], y_2[:30], y_3[:30]))
X_test = np.vstack((X_1[30:], X_2[30:], X_3[30:]))
y_test = np.hstack((y_1[30:], y_2[30:], y_3[30:]))

# OVO多类分类器
list_setosa = []
list_versicolor = []
list_virginica = []
for i in range(len(y_train)):
    if y_train[i] == "setosa":
        list_setosa.append(X_train[i])
    elif y_train[i] == "versicolor":
        list_versicolor.append(X_train[i])
    else:
        list_virginica.append(X_train[i])

arr_setosa = np.array(list_setosa)
arr_versicolor = np.array(list_versicolor)
arr_virginica = np.array(list_virginica)
# setosa(1) and versicolor(-1)
w1 = PLA(np.vstack((arr_setosa, arr_versicolor)), np.hstack((np.ones(arr_setosa.shape[0]), -np.ones(arr_versicolor.shape[0]))), 1000)
# setosa(1) and virginica(-1)
w2 = PLA(np.vstack((arr_setosa, arr_virginica)), np.hstack((np.ones(arr_setosa.shape[0]), -np.ones(arr_virginica.shape[0]))), 1000)
# versicolor(1) and virginica(-1) (线性不可分)
w3 = PLA(np.vstack((arr_versicolor, arr_virginica)), np.hstack((np.ones(arr_versicolor.shape[0]), -np.ones(arr_virginica.shape[0]))), 1000)
# test
right_num = 0
X_test_ag = np.insert(X_test, 0, np.ones(X_test.shape[0]), 1)
for i in range(len(X_test)):
    setosa_time, virginica_time, versicolor_time = 0, 0, 0
    if np.sign(np.dot(X_test_ag[i], w1)) == 1:
        setosa_time += 1
    elif np.sign(np.dot(X_test_ag[i], w1)) == -1:
        versicolor_time += 1
    if np.sign(np.dot(X_test_ag[i], w2)) == 1:
        setosa_time += 1
    elif np.sign(np.dot(X_test_ag[i], w2)) == -1:
        virginica_time += 1
    if np.sign(np.dot(X_test_ag[i], w3)) == 1:
        versicolor_time += 1
    elif np.sign(np.dot(X_test_ag[i], w3)) == -1:
        virginica_time += 1
    if setosa_time == max(setosa_time, virginica_time, versicolor_time) and y_test[i] == 'setosa':
        right_num += 1
    elif virginica_time == max(setosa_time, virginica_time, versicolor_time) and y_test[i] == 'virginica':
        right_num += 1
    elif versicolor_time == max(setosa_time, virginica_time, versicolor_time) and y_test[i] == 'versicolor':
        right_num += 1

print("OVO accuracy:", 100 * right_num / len(y_test), "%\n")


# SoftMax多分类
X_train_ag = np.insert(X_train, 0, np.ones(X_train.shape[0]), 1)
w1 = np.zeros(X_train_ag.shape[1])
w2 = np.zeros(X_train_ag.shape[1])
w3 = np.zeros(X_train_ag.shape[1])
# train
epoch = 1000
for i in range(epoch):
    for j in range(len(y_train)):
        s1 = np.dot(w1, X_train_ag[j])
        s2 = np.dot(w2, X_train_ag[j])
        s3 = np.dot(w3, X_train_ag[j])
        y1_pre = math.exp(s1) / (math.exp(s1) + math.exp(s2) + math.exp(s3))
        y2_pre = math.exp(s2) / (math.exp(s1) + math.exp(s2) + math.exp(s3))
        y3_pre = math.exp(s3) / (math.exp(s1) + math.exp(s2) + math.exp(s3))
        if y_train[j] == 'setosa':
            w1 -= 0.1 * (y1_pre - 1) * X_train_ag[j]
        else:
            w1 -= 0.1 * y1_pre * X_train_ag[j]
        if y_train[j] == 'versicolor':
            w2 -= 0.1 * (y2_pre - 1) * X_train_ag[j]
        else:
            w2 -= 0.1 * y2_pre * X_train_ag[j]
        if y_train[j] == 'virginica':
            w3 -= 0.1 * (y3_pre - 1) * X_train_ag[j]
        else:
            w3 -= 0.1 * y3_pre * X_train_ag[j]

# test
right_num = 0
for i in range(len(y_test)):
    s1 = np.dot(w1, X_test_ag[i])
    s2 = np.dot(w2, X_test_ag[i])
    s3 = np.dot(w3, X_test_ag[i])
    if s1 == max(s1, s2, s3) and y_test[i] == 'setosa':
        right_num += 1
    elif s2 == max(s1, s2, s3) and y_test[i] == 'versicolor':
        right_num += 1
    elif s3 == max(s1, s2, s3) and y_test[i] == 'virginica':
        right_num += 1

print("Softmax accuracy:", 100 * right_num / len(y_test), "%\n")