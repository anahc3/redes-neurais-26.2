"""Exercise 1 - Perceptron on linearly separable data."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron, step

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

N_PER_CLASS = 1000
MEAN_0, COV_0 = np.array([1.5, 1.5]), np.array([[0.5, 0.0], [0.0, 0.5]])
MEAN_1, COV_1 = np.array([5.0, 5.0]), np.array([[0.5, 0.0], [0.0, 0.5]])
COLORS = ["#4C72B0", "#DD8452"]


def generate_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """1000 samples per class from two well-separated Gaussians."""
    X0 = rng.multivariate_normal(MEAN_0, COV_0, size=N_PER_CLASS)
    X1 = rng.multivariate_normal(MEAN_1, COV_1, size=N_PER_CLASS)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(N_PER_CLASS, dtype=int), np.ones(N_PER_CLASS, dtype=int)])
    return X, y


def plot_boundary(ax_x, w, b):
    """x2 values of the decision line w0*x1 + w1*x2 + b = 0 over ax_x."""
    return -(w[0] * ax_x + b) / w[1]


X, y = generate_data(rng)

# --- Figure 1: raw scatter ---
plt.figure(figsize=(7, 6))
for cls, color in enumerate(COLORS):
    mask = y == cls
    plt.scatter(X[mask, 0], X[mask, 1], s=15, alpha=0.6, color=color, label=f"Class {cls}")
plt.title("Figure 1 - Separable data (Exercise 1)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_separable_scatter.png", dpi=150)
plt.close()

# --- Train with eta = 0.01 ---
w_init = rng.normal(0, 0.01, size=2)
model = Perceptron(eta=0.01, max_epochs=100, w_init=w_init, b_init=0.0)
result = model.fit(X, y)

print(f"[eta=0.01] w={result.w}, b={result.b:.4f}")
print(f"[eta=0.01] epochs_ran={result.epochs_ran}, converged={result.converged}")
print(f"[eta=0.01] final accuracy={result.acc_history[-1]:.4f}")

# --- Figure 2: decision boundary + misclassified points ---
yhat = step(X @ result.w + result.b)
misclassified = yhat != y
x_span = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])

plt.figure(figsize=(7, 6))
for cls, color in enumerate(COLORS):
    mask = (y == cls) & ~misclassified
    plt.scatter(X[mask, 0], X[mask, 1], s=15, alpha=0.5, color=color, label=f"Class {cls}")
plt.scatter(
    X[misclassified, 0], X[misclassified, 1],
    s=40, facecolors="none", edgecolors="red", linewidths=1.5, label="Misclassified",
)
plt.plot(x_span, plot_boundary(x_span, result.w, result.b), "k--", label="Decision boundary")
plt.title("Figure 2 - Decision boundary & misclassified points (eta=0.01)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_separable_boundary.png", dpi=150)
plt.close()

# --- Figure 3: accuracy vs epoch ---
plt.figure(figsize=(7, 5))
plt.plot(range(1, len(result.acc_history) + 1), result.acc_history, marker="o", color="#4C72B0")
plt.title("Figure 3 - Accuracy vs. epoch (eta=0.01)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_separable_accuracy.png", dpi=150)
plt.close()

# --- Part D.2: re-run with eta = 1.0, same initial w/b, same data ---
model_eta1 = Perceptron(eta=1.0, max_epochs=100, w_init=w_init.copy(), b_init=0.0)
result_eta1 = model_eta1.fit(X, y)
cos_sim = float(
    (result.w @ result_eta1.w) / (np.linalg.norm(result.w) * np.linalg.norm(result_eta1.w))
)

print(f"[eta=1.0] w={result_eta1.w}, b={result_eta1.b:.4f}")
print(f"[eta=1.0] epochs_ran={result_eta1.epochs_ran}, final accuracy={result_eta1.acc_history[-1]:.4f}")
print(f"Cosine similarity between w/||w|| (eta=0.01 vs eta=1.0): {cos_sim:.6f}")

if __name__ == "__main__":
    pass
