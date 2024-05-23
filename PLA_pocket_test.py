import numpy as np
from pocket import pocket
from PLA import PLA
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

# get w of PLA and Pocket
w_pla = PLA(X_train, y_train, 1000)
w_pocket = pocket(X_train, y_train, 1000)
print("PLA_w:", w_pla)
print("Pocket_w:", w_pocket)

# calculate accuracy
right_num = 0
x = np.ones(X_train.shape[0]).T
X_train_augmented = np.insert(X_train, 0, x, 1)
for i in range(X_train_augmented.shape[0]):
    if np.sign(np.dot(X_train_augmented[i], w_pla)) == np.sign(y_train[i]):
        right_num += 1
print("pla train accuracy:", right_num / len(X_train) * 100, "%")

right_num = 0
x = np.ones(X_test.shape[0]).T
X_test_augmented = np.insert(X_test, 0, x, 1)
for i in range(X_test_augmented.shape[0]):
    if np.sign(np.dot(X_test_augmented[i], w_pla)) == np.sign(y_test[i]):
        right_num += 1
print("pla test accuracy:", right_num / len(X_test) * 100, "%")

right_num = 0
x = np.ones(X_train.shape[0]).T
X_train_augmented = np.insert(X_train, 0, x, 1)
for i in range(X_train_augmented.shape[0]):
    if np.sign(np.dot(X_train_augmented[i], w_pocket)) == np.sign(y_train[i]):
        right_num += 1
print("pocket train accuracy:", right_num / len(X_train) * 100, "%")

right_num = 0
x = np.ones(X_test.shape[0]).T
X_test_augmented = np.insert(X_test, 0, x, 1)
for i in range(X_test_augmented.shape[0]):
    if np.sign(np.dot(X_test_augmented[i], w_pocket)) == np.sign(y_test[i]):
        right_num += 1
print("pocket test accuracy:", right_num / len(X_test) * 100, "%")

# draw the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X_1[:, 0], X_1[:, 1], color='blue', label='+1 class')
plt.scatter(X_2[:, 0], X_2[:, 1], color='red', label='-1 class')

# draw the pla decision boundary
x_plot = np.linspace(-10, 10, 100)
y_plot_pla = -(w_pla[1] * x_plot + w_pla[0]) / w_pla[2]
plt.plot(x_plot, y_plot_pla, color='green', linestyle='--', label='PLA Decision boundary')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Data set and PLA Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()

# draw the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X_1[:, 0], X_1[:, 1], color='blue', label='+1 class')
plt.scatter(X_2[:, 0], X_2[:, 1], color='red', label='-1 class')

# draw the pocket decision boundary
x_plot = np.linspace(-10, 10, 100)
y_plot_pocket = -(w_pocket[1] * x_plot + w_pocket[0]) / w_pocket[2]
plt.plot(x_plot, y_plot_pocket, color='yellow', linestyle='--', label='Pocket Decision boundary')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Data set and Pocket Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()

