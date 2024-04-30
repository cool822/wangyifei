import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tool.activation import *
from tool.loss import *
from tool.visualize import *


class MLP():
    def __init__(self, nn_architecture, seed=99):
        self.nn_architecture = nn_architecture
        self.learning_rate = 0
        self.params_values = {}
        self.grad_values = {}
        self.memory = {}
        self.number_of_layers = len(nn_architecture)
        self.visualize_train = None
        for layer_id, layer in enumerate(nn_architecture, 1):
            layer_input_size = layer['input_dim']
            layer_output_size = layer['output_dim']
            self.params_values['W' + str(layer_id)] = np.random.randn(layer_input_size, layer_output_size) * 0.1
            self.params_values['b' + str(layer_id)] = np.random.randn(layer_output_size, 1) * 0.1

    def single_layer_forward_propagation(self, Z_pre, W, b, activation='relu'):
        Y = np.dot(W.T, Z_pre) + b
        if activation is 'relu':
            activation_func = relu
        elif activation is 'sigmoid':
            activation_func = sigmoid
        else:
            raise Exception('没有找到该激活函数：' + activation)
        return activation_func(Y), Y

    def forward(self, X):
        Z_curr = X
        for layer_id, layer in enumerate(self.nn_architecture, 1):
            Z_pre = Z_curr
            Z_curr, Y_curr = self.single_layer_forward_propagation(Z_pre, self.params_values['W' + str(layer_id)],
                                                                   self.params_values['b' + str(layer_id)],
                                                                   layer['activation'])
            self.memory['Y' + str(layer_id)] = Y_curr
            self.memory['Z' + str(layer_id)] = Z_curr
        return Z_curr

    def single_layer_backward_propagation(self, dz_curr, w, b, y, z_pre, activation='relu'):
        """
        一层反向传播
        :param dz_curr: （样本数，输出层数）
        :param w: （输入层数，输出层数）
        :param b: （输出层数，1）
        :param y: （样本数，输出层数）
        :param z_pre: （样本数，输入层数）
        :param activation:
        :return:
        """
        if activation is 'relu':
            back_activation_func = relu_backward
        elif activation is 'sigmoid':
            back_activation_func = sigmoid_backwrad
        else:
            raise Exception('没有找到该激活函数：' + activation)
        m = dz_curr.shape[1]  # 样本数
        dy = back_activation_func(dz_curr, y)
        dw = np.dot(z_pre, dy.T) / m
        db = np.sum(dy, axis=1, keepdims=True) / m
        dz_pre = np.dot(w, dy)
        return dz_pre, dw, db

    def backward(self, Y_hat, Y):
        """
        反向传播
        :param Y_hat:（样本数，1）
        :param Y: （样本数，1）
        :return:
        """
        dz_pre = binary_cross_entropy_loss_backward(Y_hat, Y)

        for layer_id, layer in reversed(list(enumerate(self.nn_architecture, 1))):
            dz_cur = dz_pre
            dz_pre, dw, db = self.single_layer_backward_propagation(dz_cur, self.params_values['W' + str(layer_id)],
                                                                    self.params_values['b' + str(layer_id)],
                                                                    self.memory['Y' + str(layer_id)],
                                                                    self.memory['Z' + str(layer_id - 1)],
                                                                    layer['activation'])
            self.grad_values['dW' + str(layer_id)] = dw
            self.grad_values['db' + str(layer_id)] = db
        self.update()

    def update(self):
        for layer_id, layer in enumerate(self.nn_architecture, 1):
            self.params_values['W' + str(layer_id)] -= self.learning_rate * self.grad_values['dW' + str(layer_id)]
            self.params_values['b' + str(layer_id)] -= self.learning_rate * self.grad_values['db' + str(layer_id)]

    def train(self, X, Y, epochs, learning_rate, verbose=False, draw_train=False):
        """
        训练
        :param X: （样本数，）
        :param Y: 二分类标签（样本数，1）
        :param epochs: 迭代次数
        :param learning_rate: 学习率
        :param verbose: 是否显示训练过程
        :param callback: 回调函数
        :return:
        """
        self.learning_rate = learning_rate
        self.memory['Z0'] = X
        loss_list = []
        accuracy_list = []
        if (draw_train):
            self.visualize_train = VisualizeTrain(self, X, Y)
        for i in range(1, epochs + 1):
            Y_hat = self.forward(X)
            loss = binary_cross_entropy_loss(Y_hat, Y)
            accuracy = self.get_accuracy_value(Y_hat, Y)
            loss_list.append(loss)
            accuracy_list.append(accuracy)
            self.backward(Y_hat, Y)
            if (i % 100 == 0):
                if (verbose):
                    print('iteration: %d, loss: %f, accuracy: %f' % (i, loss, accuracy))
                if (draw_train):
                    self.visualize_train.callback_numpy_plot(i)

    def convert_prob_to_class_binary(self, Y):
        Y_ = Y.copy()
        Y_[Y_ > 0.5] = 1
        Y_[Y_ <= 0.5] = 0
        return Y_

    def get_accuracy_value(self, Y_hat, Y):
        """
        计算准确率
        :param Y_hat:预测值（样本数，类别数）
        :param Y:真实值（样本数，类别数）
        :return:准确率
        """
        Y_hat_ = self.convert_prob_to_class_binary(Y_hat)
        return (Y_hat_ == Y).all(axis=0).mean()


if __name__ == '__main__':
    nn_architecture = [{"input_dim": 2, "output_dim": 25, "activation": "relu"},
                       {"input_dim": 25, "output_dim": 50, "activation": "relu"},
                       {"input_dim": 50, "output_dim": 50, "activation": "relu"},
                       {"input_dim": 50, "output_dim": 25, "activation": "relu"},
                       {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"}]
    mlp = MLP(nn_architecture)
    X, Y = make_moons(n_samples=1000, noise=0.2, random_state=100)
    Y = Y.reshape(Y.shape[0], 1)
    # make_plot(X, Y, "数据集")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    mlp.train(X_train.T, Y_train.T, 3000, 0.1, verbose=True, draw_train=True)
