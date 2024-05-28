import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class FullConnectLayer(object):

    """
    全连接层类别定义
    """
    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.w = np.random.normal(0, 0.01, (self.n_in, self.n_out))
        self.b = np.zeros(self.n_out)

    def forward(self, input):
        """
        前传递函数
        """
        self.input = input
        self.output = np.matmul(input, self.w) + self.b
        return self.output

    def backward(self, top_d):
        """
        后传递函数
        """
        self.d_w = np.dot(self.input.T, top_d)
        self.d_b = np.sum(top_d, axis=0)
        bottom_d = np.dot(top_d, self.w.T)
        return bottom_d

    def update(self):
        """
        对全连接层的参数进行更新
        """
        self.w = self.w - 0.01 * self.d_w
        self.b = self.b - 0.01 * self.d_b


class Relulayer(object):
    """
    Relu函数激活层
    """
    def __init__(self):
        return

    def forward(self, input):
        """
        前传递函数
        """
        self.input = input
        output = np.maximum(0, input)
        return output

    def backward(self, top_d):
        """
        后传递函数
        """
        bottom_d = top_d
        bottom_d[self.input < 0] = 0
        return bottom_d


class SoftmaxLossLayer(object):
    """
    softmax输出层
    """
    def __init__(self):
        return

    def forward(self, input):
        """
        前传递函数
        """
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.y_pred = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.y_pred

    def get_loss(self, y_ture):
        """
        计算损失
        """
        self.out_num = self.y_pred.shape[0]
        self.y_ture_onehot = np.zeros_like(self.y_pred)
        y_ture = y_ture.astype(np.int8)
        self.y_ture_onehot[np.arange(self.out_num), y_ture] = 1.0
        loss = -np.sum(np.log(self.y_pred) * self.y_ture_onehot) / self.out_num
        return loss

    def backward(self):
        """
        后传递函数
        """
        bottom_d = (self.y_pred - self.y_ture_onehot) / self.out_num
        return bottom_d


class NetWork(object):
    """
    建立一个神经网络，其结构如下
    输入->全连接层->激活层->全连接层->激活层->全连接层->输出层
    """
    def __init__(self, n_in, hidden1, hidden2, n_out):
        self.f1 = FullConnectLayer(n_in, hidden1)
        self.r1 = Relulayer()
        self.f2 = FullConnectLayer(hidden1, hidden2)
        self.r2 = Relulayer()
        self.f3 = FullConnectLayer(hidden2, n_out)
        self.soft = SoftmaxLossLayer()
        self.update_layer_list = [self.f1, self.f2, self.f3]

    def forward(self, input):
        """
        神经网络的前传递
        """
        h1 = self.f1.forward(input)
        h1 = self.r1.forward(h1)
        h2 = self.f2.forward(h1)
        h2 = self.r2.forward(h2)
        h3 = self.f3.forward(h2)
        output = self.soft.forward(h3)
        return output

    def backward(self):
        """
        神经网络的后传递
        """
        dloss = self.soft.backward()
        dh3 = self.f3.backward(dloss)
        dh2 = self.r2.backward(dh3)
        dh2 = self.f2.backward(dh2)
        dh1 = self.r1.backward(dh2)
        dh1 = self.f1.backward(dh1)

    def update(self):
        """
        神经网络的参数更新
        """
        for layer in self.update_layer_list:
            layer.update()

    def train(self, datas, y_tures):
        """
        神经网络的训练函数
        """
        num = 0
        losses = []
        # for data, y_ture in zip(datas, y_tures):
        for i in range(10000):
            self.forward(datas)
            self.soft.get_loss(y_tures)
            self.backward()
            self.update()
            num += 1
            if num % 10 == 0:
                loss = self.soft.get_loss(y_tures)
                losses.append(loss)
        return losses


data = pd.read_csv("F:\Codes and Works\python_work\Pattern_Recognition_and_Machine_Learning\实验数据集\Iris数据集\iris.csv")
data_list = data.values.tolist()
data_arr = np.array(data_list)
X = data_arr[:, 1:5]
X = X.astype(np.double)
y = data_arr[:, 5]
# print(data_arr)
# print(X)
# print(y)
X_1 = X[:50]
y_1 = np.zeros(50)
X_2 = X[50:100]
y_2 = np.ones(50)
X_3 = X[100:150]
y_3 = 2 * np.ones(50)

indices = np.arange(X_1.shape[0])
np.random.shuffle(indices)
X_1 = X_1[indices]
y_1 = y_1[indices]
X_2 = X_2[indices]
y_2 = y_2[indices]
X_3 = X_3[indices]
y_3 = y_3[indices]

X_train = np.vstack((X_1[:30], X_2[:30], X_3[:30]))
y_train = np.hstack((y_1[:30], y_2[:30], y_3[:30]))
X_test = np.vstack((X_1[30:], X_2[30:], X_3[30:]))
y_test = np.hstack((y_1[30:], y_2[30:], y_3[30:]))

network = NetWork(4, 16, 10, 3)
lossdatas = network.train(X_train, y_train)
epoch = range(0, 10000, 10)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title("Neural Network Loss vs. Epochs")
plt.plot(epoch, lossdatas)
plt.show()

test_pred = np.argmax(network.forward(X_test), axis=1)
count = 0
for i, j in zip(y_test, test_pred):
    if i == j:
        count += 1

print("Accuracy: ", count / len(y_test) * 100, "%")
