import numpy as np
import networkx as nx

from igraph.models import Sequential
from igraph.layers import Linear, GCNLayer, SoftMax
from igraph.losses import cross_entropy_loss
from igraph.preprocessing import matrix_factorization


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    This function calculates the accuracy of the model.

    :param y_true: true labels
    :param y_pred: predicted labels
    :return: accuracy
    """

    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))


def train(model,
          A_hat: np.ndarray, X: np.ndarray, y: np.ndarray,
          loss_fn, epochs: int, early_stopping_patience: int = 50,
          log_interval: int = 1):
    """
    This function trains the model on the given data.

    :param model: igraph model
    :param A_hat: normalized adjacency matrix of the graph from A_hat = D^-1/2 * A_mod * D^-1/2, where A_mod = A + I
    :param X: feature matrix
    :param y: label matrix
    :param loss_fn: igraph loss function
    :param epochs: number of epochs
    :param early_stopping_patience: early stopping patience
    :param log_interval: logging interval
    :return: trained model
    """

    min_loss = np.inf
    patience = 0

    for epoch in range(epochs):

        output = model((A_hat, X))

        loss = loss_fn(y, output)

        grad = (output - y) / y.shape[0]  # (bs, out_dim)

        model.backward(grad)

        if epoch % log_interval == 0:
            test_loss, test_acc = test(model, A_hat, X, y, loss_fn)
            print(f'Epoch: {epoch + 1},\tLoss: {loss.item()},\tTest Loss: {test_loss.item()},\tTest Acc: {test_acc}')

        if loss < min_loss:
            min_loss = loss
            patience = 0
        else:
            patience += 1

        if patience == early_stopping_patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break


def test(model,
         A_hat: np.ndarray, X: np.ndarray, y: np.ndarray,
         loss_fn) -> tuple[float, float]:
    """
    This function tests the model on the given data.

    :param model: igraph model
    :param A_hat: normalized adjacency matrix of the graph from A_hat = D^-1/2 * A_mod * D^-1/2, where A_mod = A + I
    :param X: feature matrix
    :param y: label matrix
    :param loss_fn: igraph loss function
    :return: loss value and accuracy
    """

    output = model((A_hat, X))

    loss = loss_fn(y, output)
    acc = accuracy(y, output)

    return loss, acc


def get_community_labels(g: nx.Graph) -> np.ndarray:
    """
    This function returns the community labels of the graph.
    :param g: networkx graph
    :return: community labels
    """
    communities = nx.algorithms.community.modularity_max.greedy_modularity_communities(g)
    colors = np.zeros(g.number_of_nodes())
    for i, com in enumerate(communities):
        colors[list(com)] = i

    n_classes = np.unique(colors).shape[0]
    labels = np.eye(n_classes)[colors.astype(int)]

    return labels


def run_example():
    """
    This is an example of how to use the train and test functions.
    """

    # load the graph
    g = nx.karate_club_graph()
    A = nx.to_numpy_array(g)

    # Normalize the adjacency matrix
    A_hat = matrix_factorization(A)

    # Create the feature matrix and get the labels
    X = np.eye(A.shape[0])
    y = get_community_labels(g)

    # Create the model
    model = Sequential(lr=0.01, layers=[
        GCNLayer(n_features=A.shape[0], n_outputs=16),
        GCNLayer(n_features=16, n_outputs=16),
        GCNLayer(n_features=16, n_outputs=2),
        Linear(in_dim=2, out_dim=y.shape[1]),
        SoftMax()
    ])

    # Train the model
    train(model, A_hat, X, y, cross_entropy_loss, epochs=15000, log_interval=100)


if __name__ == '__main__':
    run_example()
