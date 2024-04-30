import numpy as np


def binary_cross_entropy_loss(Y_hat, Y):

    delta = 1e-7
    loss = -(np.dot(Y, np.log(Y_hat + delta).T) + np.dot(1 - Y, np.log(1 - Y_hat + delta).T)) / len(Y)
    return np.squeeze(loss)


def binary_cross_entropy_loss_backward(Y_hat, Y):
    return -(np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))


def cross_entropy_loss(Y_hat, Y):

    m, n = Y_hat.shape
    loss = 0
    for i in range(m):
        loss += -np.dot(Y_hat[i], np.log(Y[i]))
    loss /= m
    return loss

