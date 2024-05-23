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
                img_array = np.array(img).reshape(-1)  # Flatten the image to a vector
                images.append(img_array)
                labels.append(int(label))
    return np.array(images), np.array(labels)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_loss(X, y, weights):
    logits = np.dot(X, weights)
    probs = softmax(logits)
    loss = -np.mean(np.sum(y * np.log(probs + 1e-9), axis=1))
    return loss

# 读取训练和测试数据
train_folder = 'F:\Codes and Works\python_work\Model_Recognition_and_Machine_Learning\实验数据集\MNIST\mnist_train'
test_folder = 'F:\Codes and Works\python_work\Model_Recognition_and_Machine_Learning\实验数据集\MNIST\mnist_test'

train_images, train_labels = load_images_from_folder(train_folder)
test_images, test_labels = load_images_from_folder(test_folder)

# 数据标准化
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
train_images_ag = np.insert(train_images, 0, np.ones(train_images.shape[0]), 1)
test_images_ag = np.insert(test_images, 0, np.ones(test_images.shape[0]), 1)
train_y = []
for i in train_labels:
    y = np.zeros(len(np.unique(train_labels)))
    y[i] = 1
    train_y.append(y)
train_y = np.array(train_y)
# 打印转换后的数据形状
print(train_y.shape)
print(f"Flattened training data shape: {train_images.shape}")
print(f"Flattened test data shape: {test_images.shape}")

# 生成9个均值为0，标准差为0.01的权向量，每个向量长度为784，为列向量
num_vectors = train_images_ag.shape[1]
vector_length = len(np.unique(train_labels))
w = np.random.normal(loc=.0, scale=0.01, size=(num_vectors, vector_length))

epochs = 10
batch_size = 256
learning_rate = 0.1
train_size = train_images_ag.shape[0]
test_size = test_images_ag.shape[0]
history = {'loss': [], 'train_acc': [], 'test_acc': []}

for epoch in range(epochs):
    # 打乱训练集
    indices = np.arange(train_size)
    np.random.shuffle(indices)
    train_images_ag = train_images_ag[indices]
    train_labels = train_labels[indices]
    train_y = train_y[indices]
    # 打乱测试集
    indices = np.arange(test_size)
    np.random.shuffle(indices)
    test_images_ag = test_images_ag[indices]
    test_labels = test_labels[indices]

    for start in range(0, train_size, batch_size):
        end = min(start + batch_size, train_size)
        X_batch = train_images_ag[start:end]
        y_batch = train_y[start:end]
        label_batch = train_labels[start:end]

        logits = np.dot(X_batch, w)
        probs = softmax(logits)
        grad_w = np.dot(X_batch.T, (probs - y_batch)) / batch_size

        w -= learning_rate * grad_w

    train_loss = compute_loss(train_images_ag, train_y, w)
    train_acc = np.mean(np.argmax(softmax(np.dot(train_images_ag, w)), axis=1) == train_labels)
    test_acc = np.mean(np.argmax(softmax(np.dot(test_images_ag, w)), axis=1) == test_labels)
    history['loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)

epoch = range(1, len(history['loss']) + 1)

plt.figure(figsize=(14, 5))
plt.subplot(1, 3, 1)
plt.plot(epoch, history['loss'], label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epoch, history['train_acc'], label='Train Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(epoch, history['test_acc'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()

plt.show()

# 抽取10个样本测试
pre_list = np.argmax(softmax(np.dot(test_images_ag[:10], w)), axis=1)
for i in range(10):
    print("NO", i, "预测类别为：", pre_list[i])
    print("NO", i, "实际类别为：", test_labels[i], "\n")
