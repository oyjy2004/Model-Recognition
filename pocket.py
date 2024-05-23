import numpy as np
import random
import time


def pocket(X, Y, interation):
    """
    :param X: 训练数据集，一行为一个数据
    :param Y: 目标标签值
    :param interation: 迭代次数
    :return: 权重系数w
    """
    # get start time
    start_time = time.perf_counter()

    # augmented X
    x = np.ones(X.shape[0]).T
    X = np.insert(X, 0, x, 1)

    # init w and w_last
    w = np.zeros(X.shape[1])
    w_last = w

    # start loop
    for i in range(interation):
        w_err = []
        w_last_err = []
        # get the err index of w_last
        for j in range(X.shape[0]):
            if np.sign(np.dot(w, X[j])) != np.sign(Y[j]):
                w_last_err.append(j)

        # get new w and the err index of new w
        if len(w_last_err) == 0:
            # if all right, break
            break
        index = random.choice(w_last_err)
        w_last = w
        w = w + X[index] * Y[index]
        for j in range(X.shape[0]):
            if np.sign(np.dot(w, X[j])) != np.sign(Y[j]):
                w_err.append(j)

        # compare the err num of w and w_last, get the better one
        if len(w_err) > len(w_last_err):
            w = w_last
        else:
            w = w

    # get end time
    end_time = time.perf_counter()
    print("Pocket Time:", 1000 * (end_time - start_time), "ms")

    return w
