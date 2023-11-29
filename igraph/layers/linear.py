import numpy as np
from igraph.layers.base_layer import CachedInputLayer


class Linear(CachedInputLayer):
    def __init__(self, in_dim: int, out_dim: int):
        self.W = np.random.randn(out_dim, in_dim) * 2 / in_dim
        self.b = np.random.randn(out_dim, 1) * 2 / in_dim

    def forward(self, batch: np.array) -> np.array:
        return (np.matmul(batch, self.W.T).T + self.b).T

    def backward(self, grad: np.array, x: np.array, lr: float) -> np.array:
        # calculate input error and weights error
        inp_err = np.dot(grad, self.W)
        w_err = np.dot(x.T, grad)

        # update values from gradient
        self.W -= lr * w_err
        self.b -= lr * grad

        return inp_err
