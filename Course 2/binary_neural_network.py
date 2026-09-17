"""Binary neural network example using TensorFlow for training and NumPy for inference.

This script trains a small network on a synthetic coffee-roasting dataset,
then reproduces the trained network's forward pass using only NumPy.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf             #type: ignore

def generate_coffee_data(n_samples=400, seed=2):
    """Create a two-feature binary classification dataset.

    Features:
        temperature: degrees Celsius
        duration: minutes
    """
    rng = np.random.default_rng(seed)
    temperature = rng.uniform(150.0, 285.0, n_samples)
    duration = rng.uniform(11.5, 15.5, n_samples)
    X = np.column_stack((temperature, duration))

    upper_duration = (-3.0 / 85.0) * temperature + 21.0
    y = (
        (temperature > 175.0)
        & (temperature < 260.0)
        & (duration > 12.0)
        & (duration < 15.0)
        & (duration <= upper_duration)
    ).astype(np.float32).reshape(-1, 1)

    return X.astype(np.float32), y


def standardize(X, mean=None, std=None):
    """Z-score normalize each feature."""
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
    return (X - mean) / std, mean, std


def sigmoid(z):
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def dense_numpy(a_in, W, b, activation):
    """Vectorized dense layer: activation(a_in @ W + b)."""
    return activation(a_in @ W + b)


def forward_numpy(X, W1, b1, W2, b2):
    """Run the trained two-layer network using only NumPy."""
    hidden = dense_numpy(X, W1, b1, sigmoid)
    logits = hidden @ W2 + b2
    return logits, sigmoid(logits)


def build_tensorflow_model():

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(2,)),
            tf.keras.layers.Dense(3, activation="sigmoid", name="hidden"),
            tf.keras.layers.Dense(1, activation="linear", name="output"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def plot_predictions(X, y, model, mean, std):
    """Plot data and the learned 0.5 decision boundary."""
    t_min, t_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    d_min, d_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
    tt, dd = np.meshgrid(
        np.linspace(t_min, t_max, 180),
        np.linspace(d_min, d_max, 180),
    )
    grid = np.column_stack((tt.ravel(), dd.ravel())).astype(np.float32)
    grid_norm, _, _ = standardize(grid, mean, std)
    logits = model.predict(grid_norm, verbose=0).reshape(tt.shape)
    probabilities = sigmoid(logits)

    plt.figure(figsize=(7, 5))
    plt.contour(tt, dd, probabilities, levels=[0.5])
    plt.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], marker="x", label="Good roast")
    plt.scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], marker="o", facecolors="none", label="Bad roast")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Duration (min)")
    plt.title("Binary Neural Network Decision Boundary")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    np.random.seed(7)

    X, y = generate_coffee_data()
    X_norm, mean, std = standardize(X)

    model = build_tensorflow_model()
    model.fit(X_norm, y, epochs=150, batch_size=32, verbose=0)

    loss, accuracy = model.evaluate(X_norm, y, verbose=0)
    print(f"Training loss: {loss:.4f}")
    print(f"Training accuracy: {accuracy:.3f}")

    W1, b1 = model.get_layer("hidden").get_weights()
    W2, b2 = model.get_layer("output").get_weights()

    tf_logits = model.predict(X_norm, verbose=0)
    np_logits, np_probabilities = forward_numpy(X_norm, W1, b1, W2, b2)

    print(f"Max TensorFlow/NumPy logit difference: {np.max(np.abs(tf_logits - np_logits)):.3e}")

    test_points = np.array([[200.0, 13.9], [200.0, 17.0]], dtype=np.float32)
    test_norm, _, _ = standardize(test_points, mean, std)
    _, test_probabilities = forward_numpy(test_norm, W1, b1, W2, b2)
    test_predictions = (test_probabilities >= 0.5).astype(int)

    for point, probability, prediction in zip(test_points, test_probabilities[:, 0], test_predictions[:, 0]):
        print(
            f"temperature={point[0]:.1f} C, duration={point[1]:.1f} min -> "
            f"P(good roast)={probability:.3f}, class={prediction}"
        )

    plot_predictions(X, y, model, mean, std)


if __name__ == "__main__":
    main()
