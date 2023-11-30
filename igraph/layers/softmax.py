import numpy as np
from igraph.layers.base_layer import CachedCustomLayer


class SoftMax(CachedCustomLayer):
    def forward(self, batch: np.array) -> np.array:
        exp = np.exp(batch - batch.max())
        inv_sum = 1 / exp.sum(keepdims=True)
        return exp, inv_sum, exp * inv_sum

    def backward(self, grad: np.array, cached: tuple[np.array, np.array, np.array], lr: float=None) -> np.array:
        exp, inv_sum, out = cached
        return out * (grad - np.sum(grad * exp, keepdims=True) * inv_sum)  # it's tuple

    def __call__(self, batch):
        self.cache = self.forward(batch)
        return self.cache[-1]

