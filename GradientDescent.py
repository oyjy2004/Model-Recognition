import numpy as np
import matplotlib.pyplot as plt


# 定义函数 f(x) = x * cos(0.25πx)
def f(x):
    return x * np.cos(0.25 * np.pi * x)


# 定义 f(x) 的导数
def grad_f(x):
    return np.cos(0.25 * np.pi * x) - 0.25 * np.pi * x * np.sin(0.25 * np.pi * x)


# 梯度下降法
def gradient_descent(x_init, learning_rate, iterations):
    """
    :param x_init: 初始值
    :param learning_rate: 学习率
    :param iterations: 循环次数
    :return: x的更新记录
    """
    x = x_init
    x_hist = [x]
    for i in range(iterations):
        grad = grad_f(x)
        x -= learning_rate * grad
        x_hist.append(x)
    return x_hist


# 随机梯度下降法
def noisy_gradient_descent(x_init, learning_rate, iterations):
    """
    :param x_init: 初始值
    :param learning_rate: 学习率
    :param iterations: 循环次数
    :return: x的更新记录
    """
    x = x_init
    x_hist = [x]
    for i in range(iterations):
        grad = grad_f(x) + np.random.normal(0, 1)
        x -= learning_rate * grad
        x_hist.append(x)
    return x_hist


# Adagrad
def adagrad(x_init, learning_rate, iterations):
    """
    :param x_init: 初始值
    :param learning_rate: 学习率
    :param iterations: 循环次数
    :return: x的更新记录
    """
    x = x_init
    eps = 1e-6
    G = np.zeros_like(x, dtype=float)
    x_hist = [x]
    for i in range(iterations):
        grad = grad_f(x)
        G += grad ** 2
        x -= learning_rate * grad / (np.sqrt(G) + eps)
        x_hist.append(x)
    return x_hist


# RMSProp
def rmsprop(x_init, learning_rate, iterations):
    """
    :param x_init: 初始值
    :param learning_rate: 学习率
    :param iterations: 循环次数
    :return: x的更新记录
    """
    x = x_init
    eps = 1e-6
    alpha = 0.9
    E = np.zeros_like(x, dtype=float)
    x_hist = [x]
    for i in range(iterations):
        grad = grad_f(x)
        E = alpha * E + (1 - alpha) * grad ** 2
        x -= learning_rate * grad / (np.sqrt(E) + eps)
        x_hist.append(x)
    return x_hist


# 动量法（Momentum）
def momentum(x_init, learning_rate, iterations):
    """
    :param x_init: 初始值
    :param learning_rate: 学习率
    :param iterations: 循环次数
    :return: x的更新记录
    """
    x = x_init
    lambda_ = 0.9
    v = np.zeros_like(x, dtype=float)
    x_hist = [x]
    for i in range(iterations):
        grad = grad_f(x)
        v = lambda_ * v - learning_rate * grad
        x += v
        x_hist.append(x)
    return x_hist


# Adam
def adam(x_init, learning_rate, iterations):
    """
    :param x_init: 初始值
    :param learning_rate: 学习率
    :param iterations: 循环次数
    :return: x的更新记录
    """
    x = x_init
    eps = 1e-6
    beta1 = 0.99
    beta2 = 0.999
    m = np.zeros_like(x, dtype=float)
    v = np.zeros_like(x, dtype=float)
    x_hist = [x]
    for t in range(1, iterations + 1):
        grad = grad_f(x)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        x_hist.append(x)
    return x_hist


# 绘图
def plot_results(x, y, title):
    plt.plot(x, y, marker='o')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

# 参数设置
x_init = -4
learning_rate = 0.4
iterations = 50

# 执行优化算法并绘图
optimizers = {
    "Gradient Descent": gradient_descent(x_init, learning_rate, iterations),
    "Noisy Gradient Descent": noisy_gradient_descent(x_init, learning_rate, iterations),
    "Adagrad": adagrad(x_init, learning_rate, iterations),
    "RMSProp": rmsprop(x_init, learning_rate, iterations),
    "Momentum": momentum(x_init, learning_rate, iterations),
    "Adam": adam(x_init, learning_rate, iterations)
}

for optimizer, values in optimizers.items():
    plot_results(range(iterations + 1), [f(x) for x in values], optimizer + " - Iterations: " + str(iterations))
