import numpy as np
from Fisher import Fisher
import matplotlib.pyplot as plt

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

# get train and test
split_index = int(train_rate * X.shape[0])
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# get w and s
w, s = Fisher(X_train, y_train)

# calculate accuracy
right_num = 0
for i in range(X_train.shape[0]):
    if (np.dot(X_train[i], w) > s and y_train[i] == 1) or (np.dot(X_train[i], w) < s and y_train[i] == -1):
        right_num += 1
print("Fisher train accuracy:", right_num / len(X_train) * 100, "%")

right_num = 0
for i in range(X_test.shape[0]):
    if (np.dot(X_test[i], w) > s and y_test[i] == 1) or (np.dot(X_test[i], w) < s and y_test[i] == -1):
        right_num += 1
print("Fisher test accuracy:", right_num / len(X_test) * 100, "%")

# draw the dataset
plt.figure(figsize=(8, 8))
ti = [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]
plt.xticks(ti)
plt.yticks(ti)
plt.scatter(X_1[:, 0], X_1[:, 1], color='blue', label='+1 class')
plt.scatter(X_2[:, 0], X_2[:, 1], color='red', label='-1 class')

# 绘制最佳投影方向
x_plot = np.linspace(-10, 10, 100)
y_plot = w[1] * x_plot / w[0]
plt.plot(x_plot, y_plot, color='green', linestyle='--', label='the w')

# 绘制分类阈值
x_plot = np.linspace(-10, 10, 100)
y_plot = (s - w[0] * x_plot) / w[1]
plt.plot(x_plot, y_plot, color='yellow', linestyle='--', label='the boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Fisher Linear Discriminant Analysis')

plt.show()
