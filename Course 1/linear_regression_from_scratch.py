"""Multivariable linear regression implemented from scratch with NumPy.

Based on concepts practiced in Course 1 of the Machine Learning Specialization.
This version is self-contained and does not depend on course utility files.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)


def predict(x, w, b):
    """Return the prediction for one example."""
    return np.dot(x, w) + b


def compute_cost(X, y, w, b):
    """Compute mean-squared-error cost for multivariable linear regression."""
    m = X.shape[0]
    cost = 0.0

    for i in range(m):
        prediction = np.dot(X[i], w) + b
        cost += (prediction - y[i]) ** 2

    return cost / (2 * m)


def compute_gradient(X, y, w, b):
    """Compute gradients of the linear-regression cost with respect to w and b."""
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.0

    for i in range(m):
        error = (np.dot(X[i], w) + b) - y[i]

        for j in range(n):
            dj_dw[j] += error * X[i, j]

        dj_db += error

    return dj_dw / m, dj_db / m


def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    """Fit w and b using batch gradient descent."""
    w = copy.deepcopy(w_in)
    b = b_in
    cost_history = []

    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        cost_history.append(compute_cost(X, y, w, b))

    return w, b, cost_history


def main():
    # Features: size (sq ft), bedrooms, floors, age (years)
    X_train = np.array([
        [2104, 5, 1, 45],
        [1416, 3, 2, 40],
        [852, 2, 1, 35],
    ], dtype=float)
    y_train = np.array([460, 232, 178], dtype=float)

    initial_w = np.zeros(X_train.shape[1])
    initial_b = 0.0

    w, b, cost_history = gradient_descent(
        X_train,
        y_train,
        initial_w,
        initial_b,
        alpha=5.0e-7,
        num_iters=1000,
    )

    print(f"Learned weights: {w}")
    print(f"Learned bias: {b:.3f}\n")

    for i in range(X_train.shape[0]):
        prediction = predict(X_train[i], w, b)
        print(f"Example {i + 1}: prediction={prediction:.2f}, target={y_train[i]:.2f}")

    plt.plot(cost_history)
    plt.title("Linear Regression Training")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
