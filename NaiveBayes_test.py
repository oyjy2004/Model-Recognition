from sklearn.datasets import load_digits
from NaiveBayes import NaiveBayes
import numpy as np

# 导入训练数据
digits = load_digits()
train_datas = digits.data[0:1000]
train_trues = digits.target[0:1000]
for i in range(len(train_datas)):
    train_datas[i] = np.array(train_datas[i]).flatten()

# 导入测试数据
test_datas = digits.data[1000:1500]
test_trues = digits.target[1000:1500]
for i in range(len(test_datas)):
    test_datas[i] = np.array(test_datas[i]).flatten()

# 构造朴素贝叶斯模型
model = NaiveBayes()
model.fit(train_datas, train_trues)

# 计算正确率
module_score = model.score(test_datas, test_trues)
print(module_score)
