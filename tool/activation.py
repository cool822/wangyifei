import numpy as np


def sigmoid(y):
    return 1 / (1 + np.exp(-y))


def sigmoid_backwrad(dz, y):
    sig = sigmoid(y)
    return dz * sig * (1 - sig)


def tanh(y):
    return np.tanh(y)


def tanh_backward(dz, y):
    return dz * (1 - np.power(tanh(y), 2))


def relu(y):
    return np.maximum(0, y)


def relu_backward(dz, y):
    dy = dz.copy()
    dy[y < 0] = 0
    return dy

