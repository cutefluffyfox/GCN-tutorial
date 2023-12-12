import numpy as np
from igraph.layers import BaseLayer
from igraph.layers import GCNLayer


class Sequential:
    def __init__(self, layers: list[BaseLayer], lr: float):
        self.layers = layers
        self.lr = lr

    def forward(self, batch: np.ndarray, adj_matrix: np.ndarray = None) -> np.ndarray:
        for layer in self.layers:
            if isinstance(layer, GCNLayer):
                batch = layer(adj_matrix, batch)
            else:
                batch = layer(batch)
        return batch

    def backward(self, grad: np.ndarray):
        for layer in self.layers[::-1]:
            grad = layer.backward(grad, self.lr) if layer.cache is None else layer.backward(grad, layer.cache, self.lr)

    def __call__(self, batch: np.ndarray, adj_matrix: np.ndarray = None) -> np.ndarray:
        return self.forward(batch, adj_matrix)
