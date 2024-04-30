import numpy as np
import matplotlib.pyplot as plt
from processing import load_data

def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_pred.shape[0]

def visualize_weights(weights, shape, title="Weights Visualization"):
    """Visualize weights as images."""
    assert len(weights.shape) == 2, "Weights must be a 2D array"
    num_weights = weights.shape[1]
    num_cols = int(np.ceil(np.sqrt(num_weights)))
    num_rows = (num_weights + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
    fig.suptitle(title)

    for i, ax in enumerate(axes.flat):
        if i < num_weights:
            img = weights[:, i].reshape(shape)
            ax.imshow(img, cmap='gray', interpolation='none')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.show()

class NN:
    def __init__(self, input_size, hidden_size, output_size, activation, lambda2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        self.lambda2 = lambda2
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def forward(self, x):
        self.Z1 = np.dot(x, self.w1) + self.b1
        if self.activation == 'sigmoid':
            self.A1 = sigmoid(self.Z1)
        elif self.activation == 'relu':
            self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.w2) + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def compute_loss(self, y_true, y_pred):
        m = y_pred.shape[0]
        cross_entropy_loss = cross_entropy(y_true, y_pred)
        l2_penalty = (self.lambda2 / (2 * m)) * (np.sum(np.square(self.w1)) + np.sum(np.square(self.w2)))
        return cross_entropy_loss + l2_penalty

    def backward(self, x, y, learning_rate):
        m = y.shape[0]
        output_gap = self.A2 - y
        dw2 = np.dot(self.A1.T, output_gap) / m
        db2 = np.sum(output_gap, axis=0, keepdims=True)
        hidden_gap = np.dot(output_gap, self.w2.T) * (sigmoid_derivative(self.Z1) if self.activation == 'sigmoid' else relu_derivative(self.Z1))
        dw1 = np.dot(x.T, hidden_gap) / m
        db1 = np.sum(hidden_gap, axis=0, keepdims=True)
        self.w2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.w1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1

    def train(self, X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size):
        num_samples = X_train.shape[0]
        num_batches = num_samples // batch_size

        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                y_pred = self.forward(X_batch)
                loss = self.compute_loss(y_batch, y_pred)
                self.backward(X_batch, y_batch, learning_rate)

            accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
            self.train_losses.append(loss)
            self.train_accuracies.append(accuracy)

            val_pred = self.forward(X_val)
            val_loss = self.compute_loss(y_val, val_pred)
            val_accuracy = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            print(
                f'Epoch: {epoch}, Train Loss: {loss:.4f}, Train Accuracy: {accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Training Accuracy')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def visualize_layer_weights(self, layer_index):
        if layer_index == 'w1':
            weights = self.w1
            shape = (28, 28)  # Assuming input is 28x28 images
            title = "Visualization of First Layer Weights"
        elif layer_index == 'w2':
            weights = self.w2
            shape = (self.hidden_size, self.output_size)
            title = "Visualization of Second Layer Weights"
        elif layer_index == 'b1':
            weights = self.b1
            shape = (1, self.hidden_size)
            title = "Visualization of First Layer Bias"
        elif layer_index == 'b2':
            weights = self.b2
            shape = (1, self.output_size)
            title = "Visualization of Second Layer Bias"
        else:
            raise ValueError("Invalid layer index")

        visualize_weights(weights, shape, title)

if __name__ == '__main__':
    train_images, train_labels, val_images, val_labels = load_data()
    input_size = 28 * 28
    hidden_size = 512
    output_size = 10
    activation = 'relu'
    lambda2 = 0.01
    learning_rate = 0.01
    batch_size = 256
    epochs = 30

    train_images = train_images.reshape(-1, input_size) / 255.0
    val_images = val_images.reshape(-1, input_size) / 255.0
    train_labels = one_hot_encode(train_labels, num_classes=10)
    val_labels = one_hot_encode(val_labels, num_classes=10)

    nn = NN(input_size, hidden_size, output_size, activation, lambda2)
    nn.train(train_images, train_labels, val_images, val_labels, epochs, learning_rate, batch_size)
    nn.plot_metrics()


    nn.visualize_layer_weights('w1')
    nn.visualize_layer_weights('w2')
    nn.visualize_layer_weights('b1')
    nn.visualize_layer_weights('b2')



    





