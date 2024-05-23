import numpy as np
from qpsolvers import solve_qp

def Primal_SVM(X, Y):
    """
    :param X:
    :param Y:
    :return: w and b
    """
    d = X.shape[1]
    n = X.shape[0]
    X_ap = np.insert(X, 0, np.ones(n).T, 1)

    q = np.zeros((d + 1, d + 1))
    q[1:, 1:] = np.eye(d)
    p = np.zeros(d + 1).T
    g = np.multiply(-Y.T, X_ap.T).T
    g = g.astype(np.double)
    h = -np.ones(n).T
    solution = solve_qp(q, p, g, h, solver='cvxopt')
    w = solution[1:]
    b = solution[0]
    return w, b


def Dual_SVM(Z, Y):
    """
    :param Z:
    :param Y:
    :return: alpha
    """
    n = Z.shape[0]

    q = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            q[i][j] = Y[i] * Y[j] * np.dot(Z[i], Z[j])
    p = -np.ones(n).T
    g = -np.eye(n)
    g = g.astype(np.double)
    h = np.zeros(n).T

    alpha = solve_qp(q, p, g, h, solver='cvxopt')
    return alpha


def Kernel_SVM(kernel, X, Y):
    """
    :param kernel: 核函数
    :param X:
    :param Y:
    :return: alpha
    """
    n = X.shape[0]

    q = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            q[i][j] = Y[i] * Y[j] * kernel(X[i], X[j])
    p = -np.ones(n).T
    g = -np.eye(n)
    g = g.astype(np.double)
    h = np.zeros(n).T

    alpha = solve_qp(q, p, g, h, solver='cvxopt')
    return alpha

def Soft_Margin_Dual_ernel_SVM(kernel, X, Y, c):
    """
    :param kernel:
    :param X:
    :param Y:
    :param c:
    :return:
    """
    n = X.shape[0]

    q = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            q[i][j] = Y[i] * Y[j] * kernel(X[i], X[j])
    p = -np.ones(n).T
    g1 = -np.eye(n)
    g2 = np.eye(n)
    g = np.concatenate((g1, g2), axis=0)
    g = g.astype(np.double)
    h1 = np.zeros(n).T
    h2 = c * np.ones(n).T
    h = np.concatenate((h1, h2), axis=0)
    alpha = solve_qp(q, p, g, h, solver='cvxopt')
    return alpha
