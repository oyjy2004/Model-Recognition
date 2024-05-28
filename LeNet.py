import numpy as np
from scipy import signal

class LeNet:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        """
        :param input_shape: 数据维数
        :param num_classes: 数据类别数
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights, self.biases = self.init_weights()

    def init_weights(self):
        weights = {
            'conv1': np.random.randn(5, 5, 1, 6) * 0.01,
            'conv2': np.random.randn(5, 5, 6, 16) * 0.01,
            'fc1': np.random.randn(400, 120) * 0.01,
            'fc2': np.random.randn(120, 84) * 0.01,
            'fc3': np.random.randn(84, self.num_classes) * 0.01
        }
        biases = {
            'conv1': np.zeros((6,)),
            'conv2': np.zeros((16,)),
            'fc1': np.zeros((120,)),
            'fc2': np.zeros((84,)),
            'fc3': np.zeros((self.num_classes,))
        }
        return weights, biases

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        sigmoid_x = self.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def conv2d(self, X, filters, stride, padding):
        """
        :param X: (batch_size, height, width, channels)
        :param filters: (height, width, channels, num)
        :param stride:
        :param padding:
        :return:
        """
        batch_size, in_height, in_width, in_channels = X.shape
        filter_height, filter_width, _, num_filters = filters.shape

        if padding > 0:
            X = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant')

        out_height = (in_height - filter_height + 2 * padding) // stride + 1
        out_width = (in_width - filter_width + 2 * padding) // stride + 1

        out = np.zeros((batch_size, out_height, out_width, num_filters))

        for i in range(0, out_height):
            for j in range(0, out_width):
                X_slice = X[:, i * stride:i * stride + filter_height, j * stride:j * stride + filter_width, :]
                for m in range(batch_size):
                    for k in range(num_filters):
                        out[m, i, j, k] = np.sum(X_slice[m] * filters[:, :, :, k], axis=(0, 1, 2))

        return out

    def avg_pool(self, X, pool_size, stride):
        batch_size, in_height, in_width, in_channels = X.shape

        out_height = (in_height - pool_size) // stride + 1
        out_width = (in_width - pool_size) // stride + 1

        out = np.zeros((batch_size, out_height, out_width, in_channels))

        for i in range(0, out_height):
            for j in range(0, out_width):
                X_slice = X[:, i * stride:i * stride + pool_size, j * stride:j * stride + pool_size, :]
                out[:, i, j, :] = np.mean(X_slice, axis=(1, 2))

        return out

    def forward_propagation(self, X):
        cache = {}

        # Convolutional layer 1
        conv1_out = self.conv2d(X, self.weights['conv1'], stride=1, padding=2) + self.biases['conv1']
        conv1_out = self.sigmoid(conv1_out)
        cache['conv1'] = conv1_out
        print(cache['conv1'].shape)

        # Average pooling layer 1
        pool1_out = self.avg_pool(conv1_out, pool_size=2, stride=2)
        cache['pool1'] = pool1_out
        print(cache['pool1'].shape)

        # Convolutional layer 2
        conv2_out = self.conv2d(pool1_out, self.weights['conv2'], stride=1, padding=0) + self.biases['conv2']
        conv2_out = self.sigmoid(conv2_out)
        cache['conv2'] = conv2_out
        print(cache['conv2'].shape)

        # Average pooling layer 2
        pool2_out = self.avg_pool(conv2_out, pool_size=2, stride=2)
        cache['pool2'] = pool2_out
        print(cache['pool2'].shape)

        # Flatten layer
        flatten_out = pool2_out.reshape(pool2_out.shape[0], -1)
        cache['flatten'] = flatten_out
        print(cache['flatten'].shape)

        # Fully connected layer 1
        fc1_out = np.dot(flatten_out, self.weights['fc1']) + self.biases['fc1']
        fc1_out = self.sigmoid(fc1_out)
        cache['fc1'] = fc1_out
        print(cache['fc1'].shape)

        # Fully connected layer 2
        fc2_out = np.dot(fc1_out, self.weights['fc2']) + self.biases['fc2']
        fc2_out = self.sigmoid(fc2_out)
        cache['fc2'] = fc2_out
        print(cache['fc2'].shape)

        # Output layer (softmax)
        output_out = np.dot(fc2_out, self.weights['fc3']) + self.biases['fc3']
        output_out = self.softmax(output_out)
        cache['output'] = output_out

        return cache

    def cross_entropy_loss(self, output, y):
        m = y.shape[0]
        # Avoid numerical instability by clipping values in output
        output = np.clip(output, 1e-15, 1 - 1e-15)
        # Compute cross-entropy loss
        loss = -np.sum(y * np.log(output)) / m
        return loss

    def backward_propagation(self, X, y, learning_rate):
        m = X.shape[0]
        cache = self.forward_propagation(X)

        # Compute gradients for the output layer
        dZ3 = cache['output'] - y
        dW3 = np.dot(cache['fc2'].T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m

        # Backpropagate through the fully connected layers
        dA2 = np.dot(dZ3, self.weights['fc3'].T)
        dZ2 = dA2 * self.sigmoid_derivative(cache['fc2'])
        dW2 = np.dot(cache['fc1'].T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.weights['fc2'].T)
        dZ1 = dA1 * self.sigmoid_derivative(cache['fc1'])
        dW1 = np.dot(cache['flatten'].T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Backpropagate through the convolutional layers
        dA_pool1 = np.dot(dZ1, self.weights['fc1'].T).reshape(cache['pool2'].shape)
        dP2 = dA_pool1 / (2 * 2)
        dP2_reshaped = np.repeat(np.repeat(dP2, 2, axis=1), 2, axis=2)
        dZ_conv2 = dP2_reshaped * self.sigmoid_derivative(cache['conv2'])
        dW_conv2 = np.zeros(self.weights['conv2'].shape)
        for i in range(self.weights['conv2'].shape[0]):
            for j in range(self.weights['conv2'].shape[1]):
                for k in range(self.weights['conv2'].shape[2]):
                    for l in range(self.weights['conv2'].shape[3]):
                        dW_conv2[i, j, k, l] = np.sum(
                            dZ_conv2[:, :, :, l] * cache['pool1'][:, i:i + dZ_conv2.shape[1], j:j + dZ_conv2.shape[2], k]) / m
        db_conv2 = np.sum(dZ_conv2, axis=(0, 1, 2), keepdims=True) / m

        dA_pool0 = np.zeros(cache['pool1'].shape)
        for i in range(m):
            for f in range(self.weights['conv2'].shape[0]):
                # 将四维数组切片转换为二维数组
                dZ_slice = dZ_conv2[i, :, :, f].reshape(-1, 1)
                weights_slice = self.weights['conv2'][f, :, :, :].reshape(-1, 1)
                # 执行转置卷积操作
                dA_pool0[i] += signal.convolve2d(dZ_slice, weights_slice, mode='full').reshape(cache['pool1'].shape[1:])
        dP1 = dA_pool0 / (2 * 2)
        dP1_reshaped = np.repeat(np.repeat(dP1, 2, axis=1), 2, axis=2)
        dZ_conv1 = dP1_reshaped * self.sigmoid_derivative(cache['conv1'])
        dW_conv1 = np.zeros(self.weights['conv1'].shape)
        for i in range(self.weights['conv1'].shape[0]):
            for j in range(self.weights['conv1'].shape[1]):
                for k in range(self.weights['conv1'].shape[2]):
                    for l in range(self.weights['conv1'].shape[3]):
                        dW_conv1[i, j, k, l] = np.sum(
                            dZ_conv1[:, :, :, l] * X[:, i:i + dZ_conv1.shape[1], j:j + dZ_conv1.shape[2], k]) / m
        db_conv1 = np.sum(dZ_conv1, axis=(0, 1, 2), keepdims=True) / m

        # Update weights and biases
        self.weights['conv1'] -= learning_rate * dW_conv1
        self.biases['conv1'] -= learning_rate * db_conv1
        self.weights['conv2'] -= learning_rate * dW_conv2
        self.biases['conv2'] -= learning_rate * db_conv2
        self.weights['fc1'] -= learning_rate * dW1
        self.biases['fc1'] -= learning_rate * db1
        self.weights['fc2'] -= learning_rate * dW2
        self.biases['fc2'] -= learning_rate * db2
        self.weights['fc3'] -= learning_rate * dW3
        self.biases['fc3'] -= learning_rate * db3


    def compute_accuracy(self, X, y):
        # Forward propagation to get predictions
        cache = self.forward_propagation(X)
        predictions = np.argmax(cache['output'], axis=1)
        true_labels = np.argmax(y, axis=1)

        # Compute accuracy
        accuracy = np.mean(predictions == true_labels)
        return accuracy

    def train_and_test(self, X_train, y_train, X_test, y_test, epochs, batch_size, learning_rate):
        train_size = len(X_train)
        test_size = len(X_test)
        num_batches = len(X_train) // batch_size
        loss_history = []
        train_accuracy_history = []
        test_accuracy_history = []

        for epoch in range(epochs):
            epoch_loss = 0
            # 打乱训练集
            indices = np.arange(train_size)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
            # 打乱测试集
            indices = np.arange(test_size)
            np.random.shuffle(indices)
            X_test = X_test[indices]
            y_test = y_test[indices]

            for batch in range(num_batches):
                start_index = batch * batch_size
                end_index = (batch + 1) * batch_size
                X_batch = X_train[start_index:end_index]
                y_batch = y_train[start_index:end_index]

                # Forward propagation
                cache = self.forward_propagation(X_batch)

                # Compute loss
                loss = self.cross_entropy_loss(cache['output'], y_batch)
                epoch_loss += loss

                # Backward propagation
                self.backward_propagation(X_batch, y_batch, learning_rate)

            # Compute average loss for the epoch
            epoch_loss /= num_batches
            loss_history.append(epoch_loss)

            # Compute training accuracy for the epoch
            train_accuracy = self.compute_accuracy(X_train, y_train)
            train_accuracy_history.append(train_accuracy)

            # Compute test accuracy for the epoch (assuming you have a separate test set)
            test_accuracy = self.compute_accuracy(X_test, y_test)
            test_accuracy_history.append(test_accuracy)

            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        return loss_history, train_accuracy_history, test_accuracy_history
