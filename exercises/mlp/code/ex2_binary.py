"""Exercise 2 - Binary classification: linear baseline vs. from-scratch MLP."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlp import MLP

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

CENTERS = [[0.0, 0.0], [3.0, 3.0], [-3.0, -3.0]]
N_SAMPLES = [500, 250, 250]
COLORS = ["#4C72B0", "#DD8452"]

X, centers_label = make_blobs(
    n_samples=N_SAMPLES, centers=CENTERS, cluster_std=1.2, random_state=42
)
y = (centers_label > 0).astype(int)  # class 0: center [0,0] (500 pts); class 1: the other two clusters (250+250)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Figure 1: scatter of all 1000 points colored by y ---
plt.figure(figsize=(7, 6))
for cls, color in enumerate(COLORS):
    mask = y == cls
    plt.scatter(X[mask, 0], X[mask, 1], s=15, alpha=0.6, color=color, label=f"Class {cls}")
plt.title("Figure 1 - Binary classification data")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_binary_scatter.png", dpi=150)
plt.close()

# --- B: linear baseline ---
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)
baseline_acc = logreg.score(X_test, y_test)
print(f"[baseline] logistic regression test accuracy = {baseline_acc:.4f}")

# --- C: from-scratch MLP ---
ARCHITECTURE = [2, 8, 1]
LR = 0.1
EPOCHS = 3000

model = MLP(layer_sizes=ARCHITECTURE, output="sigmoid", lr=LR, rng=rng)
y_train_target = y_train.reshape(-1, 1).astype(float)
loss_history = model.fit(X_train, y_train_target, epochs=EPOCHS)

y_pred = model.predict(X_test)
mlp_acc = float((y_pred == y_test).mean())
print(f"[mlp] architecture={ARCHITECTURE}, lr={LR}, epochs={EPOCHS}")
print(f"[mlp] final training loss = {loss_history[-1]:.4f}")
print(f"[mlp] test accuracy = {mlp_acc:.4f}")

# --- Figure 2: training loss vs epoch ---
plt.figure(figsize=(7, 5))
plt.plot(range(1, EPOCHS + 1), loss_history, color="#4C72B0")
plt.title("Figure 2 - MLP training loss vs. epoch (Exercise 2)")
plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy loss")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig2_binary_loss.png", dpi=150)
plt.close()

# --- Figure 3: decision boundary over test points ---
pad = 1.0
x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.column_stack([xx.ravel(), yy.ravel()])
probs = model.predict_proba(grid).reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, probs, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.5)
plt.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.5)
for cls, color in enumerate(COLORS):
    mask = y_test == cls
    plt.scatter(X_test[mask, 0], X_test[mask, 1], s=20, color=color, edgecolor="black", linewidth=0.3, label=f"Class {cls}")
plt.title("Figure 3 - MLP decision boundary over test points (Exercise 2)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_binary_boundary.png", dpi=150)
plt.close()

if __name__ == "__main__":
    pass
