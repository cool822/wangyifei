import os
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class VisualizeTrain():
    def __init__(self, model, X, Y):
        self.model = model
        self.X = X
        self.Y = Y
        GRID_X_START = -2.5
        GRID_X_END = 2.5
        GRID_Y_START = -2.5
        GRID_Y_END = 2.5
        grid = np.mgrid[GRID_X_START:GRID_X_END:500j, GRID_Y_START:GRID_Y_END:500j]
        self.grid_2d = grid.reshape(2, -1).T
        self.XX, self.YY = grid
        plt.ion()
        plt.figure(figsize=(16, 12))
        plt.subplots_adjust(left=0.20)
        plt.subplots_adjust(right=0.80)
        self.axes = plt.gca()
        self.axes.set(xlabel="$X_1$", ylabel="$X_2$")

    def callback_numpy_plot(self, index):
        plot_title = "迭代次数{:05}".format(index)
        prediction_probs = self.model.forward(np.transpose(self.grid_2d))
        prediction_probs = prediction_probs.reshape(prediction_probs.shape[1], 1)
        plt.clf()
        plt.title(plot_title, fontsize=30)
        if (self.XX is not None and self.YY is not None and prediction_probs is not None):
            plt.contourf(self.XX, self.YY, prediction_probs.reshape(self.XX.shape), levels=[0, 0.5, 1], alpha=1,
                         cmap=cm.Spectral)
            plt.contour(self.XX, self.YY, prediction_probs.reshape(self.XX.shape), levels=[.5], cmap="Greys", vmin=0,
                        vmax=.6)
        plt.scatter(self.X[0, :], self.X[1, :], c=self.Y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='black')
        plt.draw()
        plt.pause(0.01)

