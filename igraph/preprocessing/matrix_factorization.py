import numpy as np


def matrix_factorization(adjacency_matrix: np.ndarray) -> np.ndarray:
    """
    Get the normalized form of adjacency_matrix for message passing.
    The formula for resulting matrix is:
        A_hat[i,j] = 1/(sqrt(d_i * d_j)) * A_mod[i,j]
    where A_mod = adjacency_matrix + I,
    d_i and d_j are degress of connected nodes.
    """

    A_mod = adjacency_matrix + np.eye(adjacency_matrix.shape[0])  # add self connections

    D = np.zeros_like(A_mod)
    np.fill_diagonal(D, A_mod.sum(axis=1))
    # D now stores the degree of each node on the diagonal
    D_inv_root = np.linalg.inv(np.sqrt(D))  # D_inv_root[i,i] = 1 / sqrt(D[i,i])

    A_hat = D_inv_root @ A_mod @ D_inv_root

    return A_hat
