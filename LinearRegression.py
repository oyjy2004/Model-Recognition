import numpy as np
import matplotlib.pyplot as plt

def GeneralizedInverse(X, Y):
    """
    :param X: 训练数据集，一行为一个数据
    :param Y: 目标标签值
    :return: 权重系数w
    """
    # augmented X
    x = np.ones(X.shape[0]).T
    X = np.insert(X, 0, x, 1)

    # get w
    w = np.linalg.inv(X.T @ X) @ X.T @ Y
    return w


def GradientDescent(X, Y, alpha, interation):
    """
    :param X: 训练数据集，一行为一个数据
    :param Y: 目标标签值
    :param alpha: 学习率
    :param interation: 迭代次数
    :return: 权重系数w
    """
    # augmented X
    x = np.ones(X.shape[0]).T
    X = np.insert(X, 0, x, 1)

    # init w and grad
    w = np.ones(X.shape[1]).T
    grad = np.zeros(X.shape[1]).T

    # init losses and times
    losses = []
    times = []

    # start loop
    for i in range(interation):
        # get loss
        loss = 0
        for j in range(X.shape[0]):
            loss += (w.T @ X[j].T - Y[j]) ** 2
        losses.append(loss / X.shape[0])
        times.append(i)

        # get grad
        grad = np.zeros(X.shape[1]).T
        for j in range(X.shape[0]):
            grad += (w.T @ X[j].T - Y[j]) * X[j].T
        grad = 2 / X.shape[0] * grad
        if grad.all() == 0:
            break

        # update w
        w = w - alpha * grad

    # draw the change of loss
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("the change of loss")
    plt.plot(times, losses)
    plt.show()

    return w
