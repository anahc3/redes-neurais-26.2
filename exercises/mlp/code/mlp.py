"""From-scratch MLP: tanh hidden layers, sigmoid (binary) or softmax (multi-class) output.

Every gradient below is hand-derived, not autodiffed. For both supported output
heads the output-layer error signal collapses to the same expression:

    delta_L = (a_L - y) / N

- sigmoid + binary cross-entropy: dL/du = sigmoid(u) - y
- softmax + categorical cross-entropy: dL/du = softmax(u) - y_onehot

(both derivations are shown in the Exercise 3 report). From there, backprop
through a tanh hidden layer uses d/dz tanh(z) = 1 - tanh(z)**2.
"""
import numpy as np


class MLP:
    def __init__(self, layer_sizes: list[int], output: str = "sigmoid",
                 lr: float = 0.1, rng: np.random.Generator = None):
        assert output in ("sigmoid", "softmax")
        self.layer_sizes = layer_sizes
        self.output = output
        self.lr = lr
        rng = rng if rng is not None else np.random.default_rng()
        self.W = []
        self.b = []
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            std = np.sqrt(2.0 / (fan_in + fan_out))  # Glorot/Xavier normal init
            self.W.append(rng.normal(0.0, std, size=(fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))

    def forward(self, X: np.ndarray):
        """Returns activations a[0..L] (a[0]=X) and pre-activations z[1..L]."""
        a = [X]
        z = [None]
        n_layers = len(self.W)
        for l in range(n_layers):
            zl = a[-1] @ self.W[l] + self.b[l]
            if l < n_layers - 1:
                al = np.tanh(zl)
            elif self.output == "sigmoid":
                al = 1.0 / (1.0 + np.exp(-zl))
            else:  # softmax
                shifted = zl - zl.max(axis=1, keepdims=True)
                expz = np.exp(shifted)
                al = expz / expz.sum(axis=1, keepdims=True)
            z.append(zl)
            a.append(al)
        return a, z

    def loss(self, y_target: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-12
        if self.output == "sigmoid":
            return float(-np.mean(y_target * np.log(y_pred + eps) + (1 - y_target) * np.log(1 - y_pred + eps)))
        return float(-np.mean(np.sum(y_target * np.log(y_pred + eps), axis=1)))

    def backward(self, a: list, y_target: np.ndarray):
        """Full-batch gradients for every W[l], b[l], starting from delta_L = (a_L - y)/N."""
        n_samples = a[0].shape[0]
        n_layers = len(self.W)
        grads_W = [None] * n_layers
        grads_b = [None] * n_layers

        delta = (a[-1] - y_target) / n_samples
        for l in reversed(range(n_layers)):
            grads_W[l] = a[l].T @ delta
            grads_b[l] = delta.sum(axis=0)
            if l > 0:
                da = delta @ self.W[l].T
                delta = da * (1.0 - a[l] ** 2)  # d/dz tanh(z) = 1 - tanh(z)^2
        return grads_W, grads_b

    def update(self, grads_W: list, grads_b: list):
        for l in range(len(self.W)):
            self.W[l] -= self.lr * grads_W[l]
            self.b[l] -= self.lr * grads_b[l]

    def fit(self, X: np.ndarray, y_target: np.ndarray, epochs: int) -> list:
        """Full-batch gradient descent. y_target must already match the output head
        (shape (N,1) in [0,1] for sigmoid, shape (N,C) one-hot for softmax)."""
        loss_history = []
        for _ in range(epochs):
            a, _ = self.forward(X)
            loss_history.append(self.loss(y_target, a[-1]))
            grads_W, grads_b = self.backward(a, y_target)
            self.update(grads_W, grads_b)
        return loss_history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        a, _ = self.forward(X)
        return a[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        if self.output == "sigmoid":
            return (p.ravel() >= 0.5).astype(int)
        return p.argmax(axis=1)

    def n_params(self) -> int:
        return sum(w.size for w in self.W) + sum(bb.size for bb in self.b)
