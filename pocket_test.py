import numpy as np
import random

# init x and y
all_x = np.empty((10, 3))
all_x[0] = [1, 0.2, 0.7]
all_x[1] = [1, 0.3, 0.3]
all_x[2] = [1, 0.4, 0.5]
all_x[3] = [1, 0.6, 0.5]
all_x[4] = [1, 0.1, 0.4]
all_x[5] = [1, 0.4, 0.6]
all_x[6] = [1, 0.6, 0.2]
all_x[7] = [1, 0.7, 0.4]
all_x[8] = [1, 0.8, 0.6]
all_x[9] = [1, 0.7, 0.5]
all_y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])

# init w and w_last
w = np.array([0, 0, 0])
w_last = w

# start loop
for i in range(20):
    w_err = []
    w_last_err = []
    # get the err index of w_last
    for j in range(all_x.shape[0]):
        y_pre = np.dot(w, all_x[j])
        if abs(y_pre) < 1.0e-5:
            y_pre = 0
        print("the w_last:", y_pre)
        if np.sign(y_pre) != np.sign(all_y[j]):
            w_last_err.append(j)

    # get new w and the err index of new w
    index = random.choice(w_last_err)
    w_last = w
    w = w + all_y[index] * all_x[index]
    for j in range(all_x.shape[0]):
        y_pre = np.dot(w, all_x[j])
        if abs(y_pre) < 1.0e-5:
            y_pre = 0
        print("the w:", y_pre)
        if np.sign(y_pre) != np.sign(all_y[j]):
            w_err.append(j)

    # compare the err num of w and w_last
    print(len(w_err), len(w_last_err))
    if len(w_err) > len(w_last_err):
        w = w_last
    else:
        w = w
    print(w, "\n\n")
