import numpy as np
from igraph.layers.base_layer import BaseLayer


class Sequential:
    def __init__(self, layers: list[BaseLayer], lr: float):
        self.layers = layers
        self.lr = lr

    def forward(self, batch: np.array) -> np.array:
        for layer in self.layers:
            batch = layer(batch)
        return batch

    def backward(self, grad: np.array):
        for layer in self.layers[::-1]:
            grad = layer.backward(grad, self.lr) if layer.cache is None else layer.backward(grad, layer.cache, self.lr)

    def __call__(self, batch) -> np.array:
        return self.forward(batch)
