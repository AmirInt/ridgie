import numpy as np


def generate_dataset(n: int, d: int, ones: list) -> tuple:
    w = np.zeros(d)

    w[ones] = 1.0

    x = np.random.normal(size=(n, d))

    e = np.random.normal(size=(n))

    y = np.dot(x, w) + e

    return x, y