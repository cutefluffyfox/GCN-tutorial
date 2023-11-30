import numpy as np
from igraph.layers.base_layer import CachedInputLayer


class Linear(CachedInputLayer):
    def __init__(self, in_dim: int, out_dim: int):
        self.W = np.random.randn(out_dim, in_dim) * 2 / in_dim
        self.b = np.random.randn(out_dim, 1) * 2 / in_dim

    def forward(self, batch: np.array) -> np.array:
        return (np.matmul(batch, self.W.T).T + self.b).T

    # @non_batchable_method(const_idx=2)
    def backward(self, grad: np.array, x: np.array, lr: float) -> np.array:
        # calculate W and b gradients
        w_grad = np.matmul(grad.T, x)
        b_grad = np.sum(grad, axis=0, keepdims=True)

        self.W -= w_grad * lr
        self.b -= b_grad.T * lr

        # calculate output gradient
        return np.matmul(self.W.T, grad.T).T

    def update(self, w_grad, b_grad, lr):
        self.W -= w_grad * lr
        self.b -= b_grad * lr
