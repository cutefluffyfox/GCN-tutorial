import numpy as np
from igraph.layers.base_layer import CachedInputLayer


class Linear(CachedInputLayer):
    def __init__(self, in_dim: int, out_dim: int):
        sd = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = np.random.uniform(-sd, sd, size=(out_dim, in_dim))
        self.b = np.zeros((out_dim, 1))

    def forward(self, batch: np.array) -> np.array:
        # TODO: replace this double transpose with a better solution
        batch = batch[1].T
        return (np.asarray(self.W @ batch) + self.b).T

    # @non_batchable_method(const_idx=2)
    def backward(self, grad: np.array, x: np.array, lr: float) -> np.array:
        # TODO: remove this line from here
        x = x[0][1]

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
