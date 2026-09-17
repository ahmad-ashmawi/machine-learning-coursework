"""L2 regularization for linear and logistic regression.

Based on concepts practiced in Course 1 of the Machine Learning Specialization.
This version is self-contained and implements regularized cost and gradient
functions without relying on course helper files.
"""

import numpy as np

np.set_printoptions(precision=8, suppress=True)


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def compute_cost_linear_reg(X, y, w, b, lambda_=1.0):
    """Compute regularized linear-regression cost."""
    m = X.shape[0]
    predictions = X @ w + b

    data_cost = np.sum((predictions - y) ** 2) / (2 * m)
    regularization_cost = lambda_ * np.sum(w ** 2) / (2 * m)

    return data_cost + regularization_cost


def compute_cost_logistic_reg(X, y, w, b, lambda_=1.0):
    """Compute regularized logistic-regression cost."""
    m = X.shape[0]
    probabilities = sigmoid(X @ w + b)
    probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)

    data_cost = -np.sum(
        y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities)
    ) / m
    regularization_cost = lambda_ * np.sum(w ** 2) / (2 * m)

    return data_cost + regularization_cost


def compute_gradient_linear_reg(X, y, w, b, lambda_=1.0):
    """Compute gradients for L2-regularized linear regression."""
    m = X.shape[0]
    error = (X @ w + b) - y

    dj_dw = (X.T @ error) / m + (lambda_ / m) * w
    dj_db = np.sum(error) / m

    return dj_dw, dj_db


def compute_gradient_logistic_reg(X, y, w, b, lambda_=1.0):
    """Compute gradients for L2-regularized logistic regression."""
    m = X.shape[0]
    error = sigmoid(X @ w + b) - y

    dj_dw = (X.T @ error) / m + (lambda_ / m) * w
    dj_db = np.sum(error) / m

    return dj_dw, dj_db


def main():
    np.random.seed(1)

    X = np.random.rand(5, 3)
    y_linear = np.array([0.2, 0.7, 0.4, 0.9, 0.3])
    y_logistic = np.array([0, 1, 0, 1, 0])
    w = np.random.rand(X.shape[1]) - 0.5
    b = 0.5
    lambda_ = 0.7

    linear_cost = compute_cost_linear_reg(X, y_linear, w, b, lambda_)
    logistic_cost = compute_cost_logistic_reg(X, y_logistic, w, b, lambda_)

    linear_dw, linear_db = compute_gradient_linear_reg(X, y_linear, w, b, lambda_)
    logistic_dw, logistic_db = compute_gradient_logistic_reg(X, y_logistic, w, b, lambda_)

    print(f"Regularized linear cost: {linear_cost:.8f}")
    print(f"Linear gradient dw: {linear_dw}")
    print(f"Linear gradient db: {linear_db:.8f}\n")

    print(f"Regularized logistic cost: {logistic_cost:.8f}")
    print(f"Logistic gradient dw: {logistic_dw}")
    print(f"Logistic gradient db: {logistic_db:.8f}")


if __name__ == "__main__":
    main()
