import numpy as np
from igraph.layers import CachedOutputLayer


class Tanh(CachedOutputLayer):
    def forward(self, batch: np.ndarray) -> np.ndarray:
        return np.tanh(batch)

    def backward(self, grad: np.ndarray, output: np.ndarray, lr: float = None) -> np.ndarray:
        return grad * (1 - np.asarray(output) ** 2)
