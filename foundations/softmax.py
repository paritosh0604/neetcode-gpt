import numpy as np
from numpy.typing import NDArray
import math

class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        sum = 0.0
        largest = max(z)
        for i in range(0,len(z),1):
            sum = sum + math.exp(z[i] - largest)
        for i in range(0,len(z),1):
            z[i] = np.round(math.exp(z[i]-largest)/sum, 4)
        return z

