import numpy as np
import math
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        for i in range(0,len(z),1):
            z[i] = 1/(1+math.exp(-1*z[i]))
            z[i] = np.round(z[i], 5)
        return z

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        for i in range(0,len(z),1):
            z[i] = max(0, z[i])
            z[i] = np.round(z[i], 5)
        return z
