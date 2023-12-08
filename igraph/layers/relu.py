import numpy as np
from igraph.layers.base_layer import CachedCustomLayer


class ReLU(CachedCustomLayer):
    def forward(self, batch: np.array) -> np.array:
        positive = batch > 0
        return positive, batch * positive

    def backward(self, grad: np.array, positive: np.array, lr: float = None) -> np.array:
        return grad * positive

    def __call__(self, batch):
        self.cache, output = self.forward(batch)
        return output
