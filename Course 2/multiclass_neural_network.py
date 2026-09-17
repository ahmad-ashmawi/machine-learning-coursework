"""Multiclass neural network using ReLU hidden layers and softmax probabilities.

The output layer produces logits. Sparse categorical cross-entropy is configured
with from_logits=True, and softmax is applied only when probabilities are needed.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf             #type: ignore 
from tensorflow import keras        #type: ignore


def make_multiclass_data(samples_per_class=100, seed=30):
    rng = np.random.default_rng(seed)
    centers = np.array([[-5.0, 2.0], [-2.0, -2.0], [1.0, 2.0], [5.0, -2.0]])

    X_parts = []
    y_parts = []
    for class_id, center in enumerate(centers):
        X_parts.append(rng.normal(loc=center, scale=1.0, size=(samples_per_class, 2)))
        y_parts.append(np.full(samples_per_class, class_id, dtype=np.int64))

    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)

    order = rng.permutation(len(X))
    return X[order], y[order]


def relu(z):
    return np.maximum(0.0, z)


def softmax(z):
    """Numerically stable softmax for a vector or batch of vectors."""
    z = np.asarray(z, dtype=np.float64)
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def build_model(num_classes):

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(2,)),
            tf.keras.layers.Dense(25, activation="relu", name="hidden_1"),
            tf.keras.layers.Dense(15, activation="relu", name="hidden_2"),
            tf.keras.layers.Dense(num_classes, activation="linear", name="logits"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    logits = model.predict(grid, verbose=0)
    classes = np.argmax(logits, axis=1).reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, classes, alpha=0.2)
    for class_id in np.unique(y):
        mask = y == class_id
        plt.scatter(X[mask, 0], X[mask, 1], label=f"Class {class_id}", s=25)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Multiclass Neural Network Decision Regions")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():

    tf.random.set_seed(30)
    np.random.seed(30)

    X, y = make_multiclass_data()
    num_classes = len(np.unique(y))

    model = build_model(num_classes)
    model.fit(X, y, epochs=120, batch_size=32, verbose=0)

    loss, accuracy = model.evaluate(X, y, verbose=0)
    print(f"Training loss: {loss:.4f}")
    print(f"Training accuracy: {accuracy:.3f}")

    logits = model.predict(X[:5], verbose=0)
    probabilities_np = softmax(logits)
    probabilities_tf = tf.nn.softmax(logits).numpy()

    print(f"Max NumPy/TensorFlow softmax difference: {np.max(np.abs(probabilities_np - probabilities_tf)):.3e}")

    for i, (logit_vector, probability_vector) in enumerate(zip(logits, probabilities_np)):
        predicted_class = int(np.argmax(logit_vector))
        print(f"Example {i}: logits={np.round(logit_vector, 3)}")
        print(f"           probabilities={np.round(probability_vector, 3)}, class={predicted_class}")

    plot_decision_boundary(X, y, model)


if __name__ == "__main__":
    main()
