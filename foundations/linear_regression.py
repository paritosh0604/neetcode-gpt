import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        res = np.zeros(len(X))
        for i in range(0, len(X)):
            for j in range(0,len(X[i])):
                res[i] = res[i] + X[i][j] * weights[j]
            res[i] = np.round(res[i],5)
        return res


    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        error = 0.0
        for i in range(0, len(model_prediction)):
            error = error + (ground_truth[i] - model_prediction[i])*(ground_truth[i] - model_prediction[i])
        error = np.round(error/len(model_prediction),5)
        return error.item()
