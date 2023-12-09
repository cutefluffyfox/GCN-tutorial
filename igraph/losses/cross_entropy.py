import numpy as np


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return -np.mean(y_true * np.log(y_pred + 1e-9))
