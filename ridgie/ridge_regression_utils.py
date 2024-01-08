import numpy as np


def mse(
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        b: float) -> float:

    residuals = y - (np.dot(x, w) + b)
    return np.dot(residuals, residuals) / y.shape[0]


def loss(
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        b: float,
        alpha: float) -> float:
    
    return np.sum(np.square(y - (np.matmul(x, w) + b))) + C * np.linalg.norm(w) ** 2


def loss_gradient_weights(
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        b: float,
        alpha: float) -> float:

    sigma = np.sum((np.matmul(x, w) + b - y).reshape(-1, 1) * x, axis=0).transpose()

    ridge = C * w

    return 2 * (sigma + ridge)


def loss_gradient_intercept(
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        b: float) -> float:

    return 2 * np.sum(np.matmul(x, w) + b - y)


def loss_convergence(losses: list) -> float:
    return abs(losses[-1] - losses[-2])


def ridge_regression_gd(
        x: np.ndarray,
        y: np.ndarray,
        alpha: float) -> tuple:

    convergence_threshold = 4e-12

    losses = []

    w = np.random.normal(size=(x.shape[1]))
    b = np.random.normal()

    losses.append(loss(x, y, w, b, alpha))

    initial_lr = 0.01

    t = 1

    # Keep doing until convergence
    while len(losses) < 2 or loss_convergence(losses) > convergence_threshold:
        w_ = w - lr * loss_gradient_weights(x, y, w, b, alpha)
        b_ = b - lr * loss_gradient_intercept(x, y, w, b)

        last_loss = loss(x, y, w_, b_, alpha)

        if last_loss < losses[-1]: # Only update the weights if the loss has reduced
            losses.append(last_loss)
            w = w_
            b = b_
        else: # If the loss did not reduce, it's probably because of a large learning rate value
            t += 1
            lr = initial_lr / t
    
    return w, b, losses