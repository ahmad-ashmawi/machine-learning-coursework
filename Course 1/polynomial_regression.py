"""Polynomial regression and feature engineering implemented with NumPy.

Based on concepts practiced in Course 1 of the Machine Learning Specialization.
This version is self-contained and demonstrates polynomial feature generation,
z-score normalization, and gradient-descent fitting.
"""

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)


def make_polynomial_features(x, degree):
    """Create [x, x^2, ..., x^degree] features."""
    return np.column_stack([x ** power for power in range(1, degree + 1)])


def zscore_normalize_features(X):
    """Normalize each feature to zero mean and unit standard deviation."""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def compute_cost(X, y, w, b):
    m = X.shape[0]
    predictions = X @ w + b
    return np.sum((predictions - y) ** 2) / (2 * m)


def compute_gradient(X, y, w, b):
    m = X.shape[0]
    error = (X @ w + b) - y
    dj_dw = (X.T @ error) / m
    dj_db = np.sum(error) / m
    return dj_dw, dj_db


def gradient_descent(X, y, alpha=0.05, num_iters=20000):
    """Fit a linear model on the supplied feature matrix."""
    w = np.zeros(X.shape[1])
    b = 0.0
    cost_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i % 100 == 0:
            cost_history.append(compute_cost(X, y, w, b))

    return w, b, cost_history


def fit_polynomial_model(x, y, degree):
    X = make_polynomial_features(x, degree)
    X_norm, mu, sigma = zscore_normalize_features(X)
    w, b, cost_history = gradient_descent(X_norm, y)
    return w, b, mu, sigma, cost_history


def predict_polynomial(x, w, b, mu, sigma):
    degree = len(w)
    X = make_polynomial_features(x, degree)
    X_norm = (X - mu) / sigma
    return X_norm @ w + b


def main():
    # Nonlinear target used to demonstrate feature engineering.
    x = np.arange(0, 20, dtype=float)
    y = 1 + x ** 2

    # Compare a purely linear feature with polynomial features.
    linear_w, linear_b, linear_mu, linear_sigma, _ = fit_polynomial_model(x, y, degree=1)
    poly_w, poly_b, poly_mu, poly_sigma, cost_history = fit_polynomial_model(x, y, degree=3)

    y_linear = predict_polynomial(x, linear_w, linear_b, linear_mu, linear_sigma)
    y_poly = predict_polynomial(x, poly_w, poly_b, poly_mu, poly_sigma)

    print(f"Polynomial weights: {poly_w}")
    print(f"Polynomial bias: {poly_b:.3f}")
    print(f"Final cost: {cost_history[-1]:.6f}")

    plt.scatter(x, y, marker="x", label="Target")
    plt.plot(x, y_linear, label="Degree 1")
    plt.plot(x, y_poly, label="Degree 3")
    plt.title("Polynomial Feature Engineering")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
