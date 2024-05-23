import math


class NaiveBayes:
    def __init__(self):
        self.model = None

    @staticmethod  # 静态方法无需实例化, 也可实例化后调用
    # 数学期望
    def mean(X):
        """计算均值
        Param: X : list or np.ndarray

        Return: avg : float
        """
        avg = 0.0
        avg = sum(X) / float(len(X))
        return avg

    # 标准差（方差）
    def stdev(self, X):
        """计算标准差
        Param: X : list or np.ndarray

        Return: res : float
        """
        res = 0.0
        avg = self.mean(X)
        res = math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))
        return res

    # 概率密度函数
    def gaussian_probability(self, x, mean, stdev):
        """根据均值和标注差计算x符号该高斯分布的概率
        Parameters:
        x : 输入
        mean : 均值
        stdev : 标准差

        Return: res : float， x符合的概率值
        """
        res = 0.0
        if stdev == 0:
            res = 0.01
        else:
            exponent = math.exp(-(math.pow(x - mean, 2) /
                                (2 * math.pow(stdev, 2))))
            res = (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
        return res

    # 处理X_train
    def summarize(self, train_data):
        """计算每个类目下对应数据的均值和标准差
        Param: train_data : list

        Return : [mean, stdev]
        """
        summaries = [0.0, 0.0]
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        return summaries

    # 分类别求出数学期望和标准差
    def fit(self, X, y):
        labels = list(set(y))
        data = {label: [] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {
            label: self.summarize(value) for label, value in data.items()
        }
        return 'gaussianNB train done!'

    # 计算概率
    def calculate_probabilities(self, input_data):
        """计算数据在各个高斯分布下的概率
        Paramter:
        input_data : 输入数据

        Return:
        probabilities : {label : p}
        """
        # module: {0.0: [(5.0, 0.37),(3.42, 0.40)], 1.0: [(5.8, 0.449),(2.7, 0.27)]}
        # input_data: [1.1, 2.2]
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.gaussian_probability(
                    input_data[i], mean, stdev)
        return probabilities

    # 类别
    def predict(self, X_test):
        # {0.0: 2.9680340789325763e-27, 1.0: 3.5749783019849535e-26}
        label = sorted(self.calculate_probabilities(X_test).items(), key=lambda x: x[-1])[-1][0]
        return label

    # 计算得分
    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1

        return right / float(len(X_test))
