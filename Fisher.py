import numpy as np

def Fisher(X, y):
    """
    :param X: 训练数据集，一行为一个数据
    :param y: 标签值
    :return: w 权重向量 s 判别门限
    """
    X1 = X[y == 1]
    X2 = X[y == -1]

    mu1 = np.mean(X1, axis=0)
    mu2 = np.mean(X2, axis=0)

    Sw = np.dot((X1 - mu1).T, X1 - mu1) + np.dot((X2 - mu2).T, X2 - mu2)
    w = np.linalg.inv(Sw) @ (mu1 - mu2).T
    s = np.dot(w.T, mu1 + mu2) / 2

    return w, s
