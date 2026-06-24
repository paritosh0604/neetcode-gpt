import numpy as np
from numpy.typing import NDArray
import math


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        for i in range(0, len(y_pred)):
            y_pred[i] = y_pred[i] + 1e-7
        sum = 0.0
        for i in range(0, len(y_pred)):
            sum = sum + (y_true[i]*math.log(y_pred[i]) + (1-y_true[i])*math.log(1-y_pred[i]))
        return np.round(-1*(1/len(y_pred))*sum, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        for j in range(0, len(y_pred)):
            for i in range(0, len(y_pred[j])):
                y_pred[j][i] = y_pred[j][i] + 1e-7
        sum = 0.0
        for j in range(0, len(y_pred)):
            for i in range(0, len(y_pred[j])):
                sum = sum + y_true[j][i]*math.log(y_pred[j][i])
        return np.round(-1*(1/len(y_pred))*sum, 4)
