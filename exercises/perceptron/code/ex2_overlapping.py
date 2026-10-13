"""Exercise 2 - Perceptron (with pocket) on overlapping, non-separable data."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron, step

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

N_PER_CLASS = 1000
MEAN_0, COV_0 = np.array([3.0, 3.0]), np.array([[1.5, 0.0], [0.0, 1.5]])
MEAN_1, COV_1 = np.array([4.0, 4.0]), np.array([[1.5, 0.0], [0.0, 1.5]])
COLORS = ["#4C72B0", "#DD8452"]


def generate_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """1000 samples per class from two heavily overlapping Gaussians."""
    X0 = rng.multivariate_normal(MEAN_0, COV_0, size=N_PER_CLASS)
    X1 = rng.multivariate_normal(MEAN_1, COV_1, size=N_PER_CLASS)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(N_PER_CLASS, dtype=int), np.ones(N_PER_CLASS, dtype=int)])
    return X, y


def plot_boundary(ax_x, w, b):
    """x2 values of the decision line w0*x1 + w1*x2 + b = 0 over ax_x."""
    return -(w[0] * ax_x + b) / w[1]


X, y = generate_data(rng)

# --- Figure 4: raw scatter ---
plt.figure(figsize=(7, 6))
for cls, color in enumerate(COLORS):
    mask = y == cls
    plt.scatter(X[mask, 0], X[mask, 1], s=15, alpha=0.5, color=color, label=f"Class {cls}")
plt.title("Figure 4 - Overlapping data (Exercise 2)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig4_overlap_scatter.png", dpi=150)
plt.close()

# --- Train with pocket tracking (eta=0.01, 100-epoch cap, reusing Exercise 1's Perceptron) ---
w_init = rng.normal(0, 0.01, size=2)
model = Perceptron(eta=0.01, max_epochs=100, w_init=w_init, b_init=0.0)
result = model.fit(X, y, track_pocket=True)

print(f"[final]  w={result.w}, b={result.b:.4f}, accuracy={result.acc_history[-1]:.4f}")
print(f"[pocket] w={result.pocket_w}, b={result.pocket_b:.4f}, "
      f"accuracy={result.pocket_acc:.4f}, found at epoch {result.pocket_epoch}")
print(f"epochs_ran={result.epochs_ran}, converged={result.converged}")

# --- Figure 5: final vs. pocket decision boundaries + misclassified (by final weights) ---
yhat_final = step(X @ result.w + result.b)
misclassified = yhat_final != y
x_span = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])

plt.figure(figsize=(7, 6))
for cls, color in enumerate(COLORS):
    mask = (y == cls) & ~misclassified
    plt.scatter(X[mask, 0], X[mask, 1], s=15, alpha=0.4, color=color, label=f"Class {cls}")
plt.scatter(
    X[misclassified, 0], X[misclassified, 1],
    s=25, facecolors="none", edgecolors="red", linewidths=1.0, label="Misclassified (final)",
)
plt.plot(x_span, plot_boundary(x_span, result.w, result.b), "k--", label="Final boundary")
plt.plot(x_span, plot_boundary(x_span, result.pocket_w, result.pocket_b), "g-", label="Pocket boundary")
plt.title("Figure 5 - Final vs. pocket decision boundaries (Exercise 2)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig5_overlap_boundaries.png", dpi=150)
plt.close()

# --- Figure 6: current-epoch accuracy vs. pocket (best-so-far) accuracy ---
epochs = range(1, len(result.acc_history) + 1)
plt.figure(figsize=(7, 5))
plt.plot(epochs, result.acc_history, marker="o", color="#4C72B0", label="Current-epoch accuracy")
plt.plot(epochs, result.pocket_acc_history, marker="s", color="#55A868", label="Pocket (best-so-far) accuracy")
plt.title("Figure 6 - Current vs. pocket accuracy (Exercise 2)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig6_overlap_accuracy.png", dpi=150)
plt.close()

if __name__ == "__main__":
    pass
