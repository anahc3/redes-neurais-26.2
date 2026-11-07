"""Exercise 4 - Depth comparison: Exercise 3's 1-hidden-layer MLP vs. a >=2-hidden-layer MLP.

Same data, split, and training budget as Exercise 3 (seed=42 for data/split);
both architectures are trained from >=3 different weight-init seeds and the
test accuracy / final train loss are averaged, so the comparison isn't a
single lucky (or unlucky) initialization.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from mlp import MLP

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

N_CLASSES = 3
X, y = make_classification(
    n_samples=1500, n_features=4, n_informative=4, n_redundant=0, n_repeated=0,
    n_classes=N_CLASSES, n_clusters_per_class=2, class_sep=1.0, random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes))
    out[np.arange(y.shape[0]), y] = 1.0
    return out


y_train_onehot = one_hot(y_train, N_CLASSES)

LR = 0.1
EPOCHS = 4000
SEEDS = [42, 43, 44]

ARCHITECTURES = {
    "Exercise 3 (1 hidden layer)": [4, 16, 3],
    "Exercise 4 (2 hidden layers)": [4, 16, 16, 3],
}

results = {name: {"test_acc": [], "train_loss": [], "loss_history": None} for name in ARCHITECTURES}

for name, arch in ARCHITECTURES.items():
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        model = MLP(layer_sizes=arch, output="softmax", lr=LR, rng=rng)
        loss_history = model.fit(X_train, y_train_onehot, epochs=EPOCHS)
        y_pred = model.predict(X_test)
        acc = float((y_pred == y_test).mean())
        results[name]["test_acc"].append(acc)
        results[name]["train_loss"].append(loss_history[-1])
        if seed == SEEDS[0]:
            results[name]["loss_history"] = loss_history
        print(f"[{name}] seed={seed} test_acc={acc:.4f} final_train_loss={loss_history[-1]:.4f}")

print("\n--- Comparison table (mean over seeds) ---")
for name, arch in ARCHITECTURES.items():
    n_params = MLP(layer_sizes=arch, output="softmax", lr=LR).n_params()
    mean_acc = np.mean(results[name]["test_acc"])
    std_acc = np.std(results[name]["test_acc"])
    mean_loss = np.mean(results[name]["train_loss"])
    n_hidden = len(arch) - 2
    print(f"{name}: hidden_layers={n_hidden}, params={n_params}, "
          f"mean_train_loss={mean_loss:.4f}, mean_test_acc={mean_acc:.4f} (std={std_acc:.4f})")

acc3 = np.mean(results["Exercise 3 (1 hidden layer)"]["test_acc"])
acc4 = np.mean(results["Exercise 4 (2 hidden layers)"]["test_acc"])
print(f"\nAccuracy difference (Ex4 - Ex3), mean over {len(SEEDS)} seeds: {acc4 - acc3:+.4f}")

# --- Figure 6: both models' loss curves (seed=42 run) ---
plt.figure(figsize=(7, 5))
for name in ARCHITECTURES:
    plt.plot(range(1, EPOCHS + 1), results[name]["loss_history"], label=name)
plt.title("Figure 6 - Training loss vs. epoch: depth comparison (seed=42)")
plt.xlabel("Epoch")
plt.ylabel("Categorical cross-entropy loss")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig6_depth_loss.png", dpi=150)
plt.close()

if __name__ == "__main__":
    pass
