import SVM_QP
import math
import numpy as np
import matplotlib.pyplot as plt

# X = np.array([[0, 0], [2, 2], [2, 0], [3, 0]])
# Y = np.array([-1, -1, 1, 1]).T
# w = SVM_QP.Primal_SVM(X, Y)
# print(w, w.shape)

def kernel1(x1, x2):
    yita = 1
    er = 1
    y = (er + yita * np.dot(x1, x2)) ** 4
    return y

def kernel2(x1, x2):
    yita = 0.03
    y = np.exp(-yita * np.linalg.norm(x1 - x2) ** 2)
    return y


# set the parameter
train_rate = 0.8
n_samples = 200
n_features = 2

# get X_1
mean_1 = [-5, 0]
covariance_1 = np.eye(n_features)
X_1 = np.random.multivariate_normal(mean_1, covariance_1, n_samples)

# get X_2
mean_2 = [0, 5]
covariance_2 = np.eye(n_features)
X_2 = np.random.multivariate_normal(mean_2, covariance_2, n_samples)

# get y_1 and y_2
y_1 = np.ones(n_samples)
y_2 = -np.ones(n_samples)

# combine X and Y
X = np.concatenate((X_1, X_2), axis=0)
y = np.concatenate((y_1, y_2))

# then disrupt the order
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# get Z
Z = np.zeros((X.shape[0], 6))
for i in range(X.shape[0]):
    Z[i][0] = 1
    Z[i][1] = X[i][0]
    Z[i][2] = X[i][1]
    Z[i][3] = X[i][0] * X[i][0]
    Z[i][4] = X[i][0] * X[i][1]
    Z[i][5] = X[i][1] * X[i][1]

# get train and test
split_index = int(train_rate * X.shape[0])
X_train, X_test = X[:split_index], X[split_index:]
Z_train, Z_test = Z[:split_index], Z[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Primal-SVM
w1, b1 = SVM_QP.Primal_SVM(X_train, y_train)
list1 = []
for i in range(X_train.shape[0]):
    if math.fabs(y_train[i] - np.dot(X_train[i], w1) - b1) < 1.0e-6:
        list1.append(i)

print("Support Vectors Index of Primal-SVM: ", list1)
print("Train Accuracy: ", np.sum(np.sign(X_train @ w1 + b1) == y_train) / X_train.shape[0] * 100, "%")
print("Test Accuracy: ", np.sum(np.sign(X_test @ w1 + b1) == y_test) / X_test.shape[0] * 100, "%\n")
# 画图
plt.figure(figsize=(8, 6))
plt.scatter(X_1[:, 0], X_1[:, 1], color='blue', label='+1')
plt.scatter(X_2[:, 0], X_2[:, 1], color='red', label='-1')
plt.scatter(X_train[list1, 0], X_train[list1, 1],
            s=100, facecolors='none', edgecolors='k', marker='o', label='Support Vectors')

x_plot = np.linspace(-10, 10, 100)
y_plot = -(w1[0] * x_plot + b1) / w1[1]
plt.plot(x_plot, y_plot, color='green', linestyle='--', label='boundary')
y_plot = (1 - (w1[0] * x_plot + b1)) / w1[1]
plt.plot(x_plot, y_plot, color='black', linestyle='--', label='')
y_plot = (-1 - (w1[0] * x_plot + b1)) / w1[1]
plt.plot(x_plot, y_plot, color='black', linestyle='--', label='')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Data set and Boundary')
plt.legend()
plt.grid(True)
plt.show()


# Dual-SVM
alpha1 = SVM_QP.Dual_SVM(Z_train, y_train)
list2 = []
w2 = np.zeros(Z.shape[1])
for i in range(len(alpha1)):
    if math.fabs(alpha1[i]) > 1.0e-6:
        list2.append(i)
    w2 += alpha1[i] * y[i] * Z_train[i]
print("Support Vectors Index of Dual-SVM: ", list2)
b2 = y_train[list2[0]] - np.dot(w2, Z_train[list2[0]])
print("Train Accuracy: ", np.sum(np.sign(Z_train @ w2 + b2) == y_train) / Z_train.shape[0] * 100, "%")
print("Test Accuracy: ", np.sum(np.sign(Z_test @ w2 + b2) == y_test) / Z_test.shape[0] * 100, "%\n")
# 画图
plt.figure(figsize=(8, 6))
plt.scatter(X_1[:, 0], X_1[:, 1], color='blue', label='+1')
plt.scatter(X_2[:, 0], X_2[:, 1], color='red', label='-1')
plt.scatter(X_train[list2, 0], X_train[list2, 1],
            s=100, facecolors='none', edgecolors='k', marker='o', label='Support Vectors')

thex = np.linspace(-10, 10, 100)
they = np.linspace(-10, 10, 100)
thex, they = np.meshgrid(thex, they)
func1 = w2[0] + w2[1] * thex + w2[2] * they + w2[3] * thex ** 2 + w2[4] * thex * they + w2[5] * they ** 2
plt.contour(thex, they, func1, levels=[0], colors='g')
plt.contour(thex, they, func1, levels=[-1, 1], colors='k')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Data set and Boundary')
plt.legend()
plt.grid(True)
plt.show()


# Kernel-SVM
alpha2 = SVM_QP.Kernel_SVM(kernel1, X_train, y_train)
list3 = []
for i in range(len(alpha2)):
    if math.fabs(alpha2[i]) > 1.0e-6:
        list3.append(i)
print("Support Vectors Index of Kernel-SVM: ", list3)
b3 = 0
for i in range(len(list3)):
    b3 -= alpha2[list3[i]] * y_train[list3[i]] * kernel1(X_train[list3[i]], X_train[list3[0]])
b3 += y_train[list3[0]]
rightnum = 0
for i in range(X_train.shape[0]):
    y_pre = 0
    for j in list3:
        y_pre += alpha2[j] * y_train[j] * kernel1(X_train[j], X_train[i])
    y_pre += b3
    if np.sign(y_pre) == y_train[i]:
        rightnum += 1
print("Train Accuracy: ", rightnum / X_train.shape[0] * 100, "%")
rightnum = 0
for i in range(X_test.shape[0]):
    y_pre = 0
    for j in list3:
        y_pre += alpha2[j] * y_train[j] * kernel1(X_train[j], X_test[i])
    y_pre += b3
    if np.sign(y_pre) == y_test[i]:
        rightnum += 1
print("Test Accuracy: ", rightnum / X_test.shape[0] * 100, "%\n")
# 画图
plt.figure(figsize=(8, 6))
plt.scatter(X_1[:, 0], X_1[:, 1], color='blue', label='+1')
plt.scatter(X_2[:, 0], X_2[:, 1], color='red', label='-1')
plt.scatter(X_train[list3, 0], X_train[list3, 1],
            s=100, facecolors='none', edgecolors='k', marker='o', label='Support Vectors')

thex = np.linspace(-10, 10, 100)
they = np.linspace(-10, 10, 100)
thex, they = np.meshgrid(thex, they)

func2 = np.zeros_like(thex)
for i in range(func2.shape[0]):
    for k in range(func2.shape[1]):
        for j in list3:
            func2[i][k] += alpha2[j] * y_train[j] * kernel1(X_train[j], np.array([thex[i][k], they[i][k]]))
        func2[i][k] += b3

plt.contour(thex, they, func2, levels=[0], colors='g')
plt.contour(thex, they, func2, levels=[-1, 1], colors='k')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Data set and Boundary')
plt.legend()
plt.grid(True)
plt.show()


# Soft_Margin_Dual_ernel_SVM
c = 1
alpha3 = SVM_QP.Soft_Margin_Dual_ernel_SVM(kernel2, X_train, y_train, c)
list4_free = []
list4_bound = []
for i in range(len(alpha3)):
    if math.fabs(alpha3[i]) > 1.0e-6 and math.fabs(alpha3[i] - c) > 1.0e-6:
        list4_free.append(i)
    elif math.fabs(alpha3[i] - c) < 1.0e-6:
        list4_bound.append(i)
print("Free Support Vectors Index of Soft-Dual-Kernel-SVM: ", list4_free)
print("Bounded Support Vectors Index of Soft-Dual-Kernel-SVM: ", list4_bound)
b4 = 0
for i in range(len(alpha3)):
    b4 -= alpha3[i] * y_train[i] * kernel2(X_train[i], X_train[list4_free[0]])
b4 += y_train[list4_free[0]]
rightnum = 0
for i in range(X_train.shape[0]):
    y_pre = 0
    for j in range(len(alpha3)):
        y_pre += alpha3[j] * y_train[j] * kernel2(X_train[j], X_train[i])
    y_pre += b4
    if np.sign(y_pre) == y_train[i]:
        rightnum += 1
print("Train Accuracy: ", rightnum / X_train.shape[0] * 100, "%")
rightnum = 0
for i in range(X_test.shape[0]):
    y_pre = 0
    for j in range(len(alpha3)):
        y_pre += alpha3[j] * y_train[j] * kernel2(X_train[j], X_test[i])
    y_pre += b4
    if np.sign(y_pre) == y_test[i]:
        rightnum += 1
print("Test Accuracy: ", rightnum / X_test.shape[0] * 100, "%\n")
# 画图
plt.figure(figsize=(8, 6))
plt.scatter(X_1[:, 0], X_1[:, 1], color='blue', label='+1')
plt.scatter(X_2[:, 0], X_2[:, 1], color='red', label='-1')
plt.scatter(X_train[list4_free, 0], X_train[list4_free, 1],
            s=100, facecolors='none', edgecolors='k', marker='o', label='Free Support Vectors')
plt.scatter(X_train[list4_bound, 0], X_train[list4_bound, 1],
            s=100, facecolors='none', edgecolors='y', marker='o', label='Bounded Support Vectors')

thex = np.linspace(-10, 10, 100)
they = np.linspace(-10, 10, 100)
thex, they = np.meshgrid(thex, they)

func3 = np.zeros_like(thex)
for i in range(func3.shape[0]):
    for k in range(func3.shape[1]):
        for j in range(len(alpha3)):
            func3[i][k] += alpha3[j] * y_train[j] * kernel2(X_train[j], np.array([thex[i][k], they[i][k]]))
        func3[i][k] += b4

plt.contour(thex, they, func3, levels=[0], colors='g')
plt.contour(thex, they, func3, levels=[-1, 1], colors='k')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Data set and Boundary')
plt.legend()
plt.grid(True)
plt.show()
