import numpy as np
import SVM_QP
import math
import matplotlib.pyplot as plt

# 训练集数据
china_coastal_cities = np.array([
    [119.28, 26.08],  # 福州
    [121.31, 25.03],  # 台北
    [121.47, 31.23],  # 上海
    [118.06, 24.27],  # 厦门
    [121.46, 39.04],  # 大连
    [122.10, 37.50],  # 威海
    [124.23, 40.07]   # 丹东
])
japan_coastal_cities = np.array([
    [129.87, 32.75],  # 长崎
    [130.33, 31.36],  # 鹿儿岛
    [131.42, 31.91],  # 宫崎
    [130.24, 33.35],  # 福冈
    [133.33, 15.43],  # 鸟取
    [138.38, 34.98],  # 静冈
    [140.47, 36.37]   # 水户
])
X_train = np.vstack((china_coastal_cities, japan_coastal_cities))
y_train = np.concatenate((np.ones(china_coastal_cities.shape[0]), -np.ones(japan_coastal_cities.shape[0])))

# 钓鱼岛的经纬度坐标
diaoyu_island = np.array([123.47, 25.74])

# 训练
w, b = SVM_QP.Primal_SVM(X_train, y_train)
print(w, b)
# 求支撑向量
list1 = []
for i in range(X_train.shape[0]):
    if math.fabs(y_train[i] - np.dot(X_train[i], w) - b) < 1.0e-6:
        list1.append(i)
print(list1)
# 画图
plt.figure(figsize=(8, 6))
plt.scatter(diaoyu_island[0], diaoyu_island[1], color='yellow', label='diaoyudao')
plt.scatter(china_coastal_cities[:, 0], china_coastal_cities[:, 1], color='blue', label='china')
plt.scatter(japan_coastal_cities[:, 0], japan_coastal_cities[:, 1], color='red', label='japan')
plt.scatter(X_train[list1, 0], X_train[list1, 1],
            s=100, facecolors='none', edgecolors='k', marker='o', label='Support Vectors')

x_plot = np.linspace(100, 150, 100)
y_plot = -(w[0] * x_plot + b) / w[1]
plt.plot(x_plot, y_plot, color='green', linestyle='--', label='boundary')
y_plot = (1 - (w[0] * x_plot + b)) / w[1]
plt.plot(x_plot, y_plot, color='black', linestyle='--', label='')
y_plot = (-1 - (w[0] * x_plot + b)) / w[1]
plt.plot(x_plot, y_plot, color='black', linestyle='--', label='')

plt.xlabel('jing')
plt.ylabel('wei')
plt.title('Data set and Boundary')
plt.legend()
plt.grid(True)
plt.show()
# 预测钓鱼岛的类别
diaoyu_prediction = np.sign(np.dot(w, diaoyu_island) + b)
print("钓鱼岛的预测类别:", "中国" if diaoyu_prediction == 1 else "日本")


# 增加非海边城市的数据
china_cities = np.array([
    [119.28, 26.08],  # 福州
    [121.31, 25.03],  # 台北
    [121.47, 31.23],  # 上海
    [118.06, 24.27],  # 厦门
    [113.53, 29.58],  # 武汉
    [121.46, 39.04],  # 大连
    [122.10, 37.50],  # 威海
    [124.23, 40.07]   # 丹东
])
japan_cities = np.array([
    [129.87, 32.75],  # 长崎
    [130.33, 31.36],  # 鹿儿岛
    [131.42, 31.91],  # 宫崎
    [130.24, 33.35],  # 福冈
    [136.54, 35.10],  # 名古屋
    [132.27, 34.24],  # 广岛
    [133.33, 15.43],  # 鸟取
    [138.38, 34.98],  # 静冈
    [140.47, 36.37]   # 水户
])
X_train_augmented = np.vstack((china_cities, japan_cities))
y_train_augmented = np.concatenate((np.ones(china_cities.shape[0]), -np.ones(japan_cities.shape[0])))

# 训练
w, b = SVM_QP.Primal_SVM(X_train_augmented, y_train_augmented)
print(w, b)
# 求支撑向量
list2 = []
for i in range(X_train_augmented.shape[0]):
    if math.fabs(y_train_augmented[i] - np.dot(X_train_augmented[i], w) - b) < 1.0e-6:
        list2.append(i)
print(list2)
# 画图
plt.figure(figsize=(8, 6))
plt.scatter(diaoyu_island[0], diaoyu_island[1], color='yellow', label='diaoyudao')
plt.scatter(china_cities[:, 0], china_cities[:, 1], color='blue', label='china')
plt.scatter(japan_cities[:, 0], japan_cities[:, 1], color='red', label='japan')
plt.scatter(X_train_augmented[list2, 0], X_train_augmented[list2, 1],
            s=100, facecolors='none', edgecolors='k', marker='o', label='Support Vectors')

x_plot = np.linspace(100, 150, 100)
y_plot = -(w[0] * x_plot + b) / w[1]
plt.plot(x_plot, y_plot, color='green', linestyle='--', label='boundary')
y_plot = (1 - (w[0] * x_plot + b)) / w[1]
plt.plot(x_plot, y_plot, color='black', linestyle='--', label='')
y_plot = (-1 - (w[0] * x_plot + b)) / w[1]
plt.plot(x_plot, y_plot, color='black', linestyle='--', label='')

plt.xlabel('jing')
plt.ylabel('wei')
plt.title('Data set and Boundary')
plt.legend()
plt.grid(True)
plt.show()

# 预测钓鱼岛的类别
diaoyu_prediction = np.sign(np.dot(w, diaoyu_island) + b)
print("钓鱼岛的预测类别:", "中国" if diaoyu_prediction == 1 else "日本")
