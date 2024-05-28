from LeNet import LeNet
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img_array = np.array(img)
                images.append(img_array)
                labels.append(int(label))
    return np.array(images), np.array(labels)


def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]


# 读取训练和测试数据
train_folder = 'F:\Codes and Works\python_work\Pattern_Recognition_and_Machine_Learning\实验数据集\MNIST\mnist_train'
test_folder = 'F:\Codes and Works\python_work\Pattern_Recognition_and_Machine_Learning\实验数据集\MNIST\mnist_test'
X_train, y_train = load_images_from_folder(train_folder)
X_test, y_test = load_images_from_folder(test_folder)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
print(X_train.shape)
# 数据标准化
X_train = X_train.astype('float32') / 255.0
X_train = X_train.astype('float32') / 255.0
y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
# 打乱训练集
indices = np.arange(len(X_train))
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]
# 打乱测试集
indices = np.arange(len(X_test))
np.random.shuffle(indices)
X_test = X_test[indices]
y_test = y_test[indices]

lenet_model = LeNet()
loss_history, train_accuracy_history, test_accuracy_history = lenet_model.train_and_test(
    X_train, y_train, X_test, y_test, epochs=10, batch_size=256, learning_rate=0.001)

epoch = range(1, len(loss_history) + 1)

plt.figure(figsize=(14, 5))
plt.subplot(1, 3, 1)
plt.plot(epoch, loss_history, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epoch, train_accuracy_history, label='Train Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(epoch, test_accuracy_history, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()

plt.show()

