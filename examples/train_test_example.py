import numpy as np


def train(model,
          A_hat: np.array, X: np.array, y: np.array,
          optimizer, loss_fn, epochs: int,
          log_interval: int = 10):
    pass


def test(model,
         A_hat: np.array, X: np.array, y: np.array,
         loss_fn):
    pass


def run_example():
    """
    This is an example of how to use the train and test functions.
    """
    pass


if __name__ == '__main__':
    run_example()
