"""Exercise 3 - Multi-class classification: softmax + categorical cross-entropy MLP."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from mlp import MLP

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

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


# --- B: logistic-regression baseline (multinomial) ---
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
baseline_acc = logreg.score(X_test, y_test)
print(f"[baseline] logistic regression test accuracy = {baseline_acc:.4f}")

# --- C: from-scratch MLP (reuses Exercise 2's MLP class verbatim) ---
ARCHITECTURE = [4, 16, 3]
LR = 0.1
EPOCHS = 4000

model = MLP(layer_sizes=ARCHITECTURE, output="softmax", lr=LR, rng=rng)
y_train_onehot = one_hot(y_train, N_CLASSES)
loss_history = model.fit(X_train, y_train_onehot, epochs=EPOCHS)

y_pred = model.predict(X_test)
mlp_acc = float((y_pred == y_test).mean())
print(f"[mlp] architecture={ARCHITECTURE}, lr={LR}, epochs={EPOCHS}")
print(f"[mlp] final training loss = {loss_history[-1]:.4f}")
print(f"[mlp] test accuracy = {mlp_acc:.4f}")

# --- Figure 4: training loss vs epoch ---
plt.figure(figsize=(7, 5))
plt.plot(range(1, EPOCHS + 1), loss_history, color="#4C72B0")
plt.title("Figure 4 - MLP training loss vs. epoch (Exercise 3)")
plt.xlabel("Epoch")
plt.ylabel("Categorical cross-entropy loss")
plt.tight_layout()
plt.savefig(FIG_DIR / "fig4_multiclass_loss.png", dpi=150)
plt.close()

# --- Figure 5: confusion matrix on test set ---
cm = confusion_matrix(y_test, y_pred)
print(f"[mlp] confusion matrix:\n{cm}")

plt.figure(figsize=(6, 5))
im = plt.imshow(cm, cmap="Blues")
plt.colorbar(im, label="Count")
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
plt.title("Figure 5 - Confusion matrix on test set (Exercise 3)")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.xticks(range(N_CLASSES))
plt.yticks(range(N_CLASSES))
plt.tight_layout()
plt.savefig(FIG_DIR / "fig5_multiclass_confusion.png", dpi=150)
plt.close()

# --- most-confused off-diagonal pair ---
cm_offdiag = cm.copy()
np.fill_diagonal(cm_offdiag, 0)
i, j = np.unravel_index(cm_offdiag.argmax(), cm_offdiag.shape)
print(f"[mlp] most-confused pair: true={i} predicted={j} ({cm_offdiag[i, j]} points); "
      f"symmetric total true<->pred = {cm_offdiag[i, j] + cm_offdiag[j, i]}")

if __name__ == "__main__":
    pass
