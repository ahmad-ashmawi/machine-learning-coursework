"""Binary logistic regression implemented from scratch with NumPy.

Based on concepts practiced in Course 1 of the Machine Learning Specialization.
This version is self-contained and includes sigmoid activation, logistic cost,
gradient descent, prediction, and a decision-boundary visualization.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)


def sigmoid(z):
    """Compute the sigmoid function."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def compute_cost(X, y, w, b):
    """Compute binary cross-entropy cost."""
    m = X.shape[0]
    probabilities = sigmoid(X @ w + b)
    probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)

    return -np.sum(
        y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities)
    ) / m


def compute_gradient(X, y, w, b):
    """Compute gradients of the logistic-regression cost."""
    m = X.shape[0]
    error = sigmoid(X @ w + b) - y
    dj_dw = (X.T @ error) / m
    dj_db = np.sum(error) / m
    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, alpha=0.1, num_iters=10000):
    """Fit logistic-regression parameters using batch gradient descent."""
    w = copy.deepcopy(w_in)
    b = b_in
    cost_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i % 100 == 0:
            cost_history.append(compute_cost(X, y, w, b))

    return w, b, cost_history


def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


def predict(X, w, b, threshold=0.5):
    return (predict_proba(X, w, b) >= threshold).astype(int)


def plot_data_and_boundary(X, y, w, b):
    positive = y == 1
    negative = y == 0

    plt.scatter(X[positive, 0], X[positive, 1], marker="x", label="Class 1")
    plt.scatter(X[negative, 0], X[negative, 1], marker="o", facecolors="none", label="Class 0")

    x0 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
    if abs(w[1]) > 1e-12:
        x1 = -(w[0] * x0 + b) / w[1]
        plt.plot(x0, x1, label="Decision boundary")

    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.title("Logistic Regression from Scratch")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    X_train = np.array([
        [0.5, 1.5],
        [1.0, 1.0],
        [1.5, 0.5],
        [3.0, 0.5],
        [2.0, 2.0],
        [1.0, 2.5],
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    initial_w = np.zeros(X_train.shape[1])
    initial_b = 0.0

    w, b, cost_history = gradient_descent(
        X_train,
        y_train,
        initial_w,
        initial_b,
        alpha=0.1,
        num_iters=10000,
    )

    predictions = predict(X_train, w, b)
    accuracy = np.mean(predictions == y_train) * 100

    print(f"Learned weights: {w}")
    print(f"Learned bias: {b:.3f}")
    print(f"Final cost: {cost_history[-1]:.6f}")
    print(f"Training accuracy: {accuracy:.1f}%")

    plot_data_and_boundary(X_train, y_train, w, b)


if __name__ == "__main__":
    main()
