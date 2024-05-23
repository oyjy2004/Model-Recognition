import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(x):
    """
    :param x: 输入值
    :return: y 函数值
    """
    x = np.clip(x, -500, 500)
    return 1 / (1 + math.exp(-x))


def logistic(X, y, interation, learnrate):
    """
    :param X: 训练数据集，一行为一个数据
    :param y: 标签值
    :param interation: 最大迭代次数
    :param learnrate: 学习率
    :return: w 权重向量
    """
    # augmented X
    x = np.ones(X.shape[0]).T
    X = np.insert(X, 0, x, 1)

    # init w and gard
    w = np.zeros(X.shape[1])
    gard = np.zeros(X.shape[1])

    # init losses and times
    loss = 0
    losses = []
    times = []

    # start loop
    for i in range(interation):
        # get loss
        loss = 0
        for j in range(X.shape[0]):
            thex = -y[j] * np.dot(w, X[j])
            thex = np.clip(thex, -500, 500)
            loss += math.log(1 + math.exp(thex))
        losses.append(loss / X.shape[0])
        times.append(i)

        # get gard
        gard = np.zeros(X.shape[1])
        for j in range(X.shape[0]):
            gard += sigmoid(-y[j] * w @ X[j].T) * (-y[j] * X[j])
        gard /= X.shape[0]

        # update w
        w -= learnrate * gard
        if gard.all() == 0:
            break

    # draw the change of loss
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("the change of loss")
    plt.plot(times, losses)
    plt.show()

    return w
