import numpy as np
from math import exp

x = np.empty((4, 3))
x[0] = [1, 3, 0]
x[1] = [1, 3, 6]
x[2] = [1, 0, 3]
x[3] = [1, -3, 0]


y = np.array([1, 1, 2, 3])

w1 = np.zeros(3)
w2 = np.zeros(3)
w3 = np.zeros(3)

for j in range(100):
    for i in range(4):
        s1 = np.dot(x[i], w1)
        s2 = np.dot(x[i], w2)
        s3 = np.dot(x[i], w3)
        y1 = exp(s1) / (exp(s1) + exp(s2) + exp(s3))
        y2 = exp(s2) / (exp(s1) + exp(s2) + exp(s3))
        y3 = exp(s3) / (exp(s1) + exp(s2) + exp(s3))
        if y[i] == 1:
            w1 -= (y1 - 1) * x[i]
        else:
            w1 -= y1 * x[i]
        if y[i] == 2:
            w2 -= (y2 - 1) * x[i]
        else:
            w2 -= y2 * x[i]
        if y[i] == 3:
            w3 -= (y3 - 1) * x[i]
        else:
            w3 -= y3 * x[i]

    if (np.dot(x[1], w1) > np.dot(x[1], w2) and np.dot(x[1], w1) > np.dot(x[1], w3) and
        np.dot(x[2], w2) > np.dot(x[2], w1) and np.dot(x[2], w2) > np.dot(x[2], w3) and
        np.dot(x[3], w3) > np.dot(x[3], w1) and np.dot(x[3], w3) > np.dot(x[3], w2)):
        break
    print(w1, "\n", w2, "\n", w3)

print(w1, "\n", w2, "\n", w3)

