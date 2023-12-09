import numpy as np
from igraph.layers.base_layer import CachedCustomLayer


class GCNLayer(CachedCustomLayer):
    def __init__(self, n_features: int, n_outputs: int):
        self.n_features = n_features
        self.n_outputs = n_outputs

        sd = np.sqrt(6.0 / (n_features + n_outputs))
        self.W = np.random.uniform(-sd, sd, size=(n_outputs, n_features))

    def forward(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        adj_matrix is (bs, bs) adjacency matrix
        batch is (bs, D),

        where bs = "batch size" and D = input feature length
        """

        adj_matrix = batch[0]
        batch = batch[1]

        features = (adj_matrix @ batch).T  # for calculating gradients (D, bs)
        # TODO: add activation function as a parameter
        return features, adj_matrix, np.tanh((self.W @ features).T)  # (bs, h)

    def backward(self, grad: np.ndarray, cached: tuple[np.ndarray, np.ndarray, np.ndarray],
                 lr: float = None) -> np.ndarray:
        features, adj_matrix, output = cached
        batch_size = output.shape[0]

        # calculate w_grad
        tanh_grad = 1 - np.asarray(output) ** 2  # (bs, out_dim)
        feature_grad = np.multiply(grad, tanh_grad)  # (bs, out_dim) *element_wise* (bs, out_dim)
        w_grad = np.asarray(feature_grad.T @ features.T) / batch_size  # (out_dim, bs)*(bs, D) -> (out_dim, D)

        self.W -= w_grad * lr

        # calculate output gradient
        return adj_matrix @ feature_grad @ self.W  # (bs, bs)*(bs, out_dim)*(out_dim, in_dim) = (bs, in_dim)

    def __call__(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.cache = self.forward(batch)
        return self.cache[-2:]
