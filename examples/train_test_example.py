import numpy as np


def train(model,
          A_hat: np.array, X: np.array, y: np.array,
          optimizer, loss_fn, epochs: int,
          log_interval: int = 10):
    """
    This function trains the model on the given data.

    :param model: igraph model
    :param A_hat: normalized adjacency matrix of the graph from A_hat = D^-1/2 * A_mod * D^-1/2, where A_mod = A + I
    :param X: feature matrix
    :param y: label matrix
    :param optimizer: igraph optimizer
    :param loss_fn: igraph loss function
    :param epochs: number of epochs
    :param log_interval: logging interval
    :return: trained model
    """

    # Some preliminary code ( high probability of being wrong )
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        output = model(A_hat, X)

        loss = loss_fn(output, y)
        loss.backward()

        optimizer.step()

        if epoch % log_interval == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

    return model


def test(model,
         A_hat: np.array, X: np.array, y: np.array,
         loss_fn):
    """
    This function tests the model on the given data.

    :param model: igraph model
    :param A_hat: normalized adjacency matrix of the graph from A_hat = D^-1/2 * A_mod * D^-1/2, where A_mod = A + I
    :param X: feature matrix
    :param y: label matrix
    :param loss_fn: igraph loss function
    :return: loss value and accuracy
    """

    # Some preliminary code ( high probability of being wrong )
    model.eval()
    output = model(A_hat, X)

    loss = loss_fn(output, y)
    acc = accuracy(output, y)

    return loss, acc


def run_example():
    """
    This is an example of how to use the train and test functions.
    """

    pass


if __name__ == '__main__':
    run_example()
