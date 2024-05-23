import numpy as np
import time


def PLA(X, Y, maxinteration):
    """
    :param X: 训练数据集，一行为一个数据
    :param Y: 目标标签
    :param maxinteration:线性不可分时最大迭代次数
    :return: 权向量w
    """
    # get start time
    start_time = time.perf_counter()

    # augmented X
    x = np.ones(X.shape[0]).T
    X = np.insert(X, 0, x, 1)

    # init w
    w = np.zeros(X.shape[1])

    # start loop
    for i in range(maxinteration):
        err_index_list = []
        for i in range(X.shape[0]):
            if np.sign(np.dot(X[i], w)) != np.sign(Y[i]):
                err_index_list.append(i)
                w = w + Y[i] * X[i]

        if len(err_index_list) == 0:
            break

    # get end time
    end_time = time.perf_counter()
    print("PLA Time:", 1000 * (end_time - start_time), "ms")

    return w
