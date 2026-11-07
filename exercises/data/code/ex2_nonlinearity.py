"""Exercise 2 - Non-linearity in higher dimensions: Gaussians vs. concentric shells."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
N = 500

# --- Dataset I: two multivariate Gaussians ---
mu_A = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
Sigma_A = np.array(
    [
        [1.0, 0.8, 0.1, 0.0, 0.0],
        [0.8, 1.0, 0.3, 0.0, 0.0],
        [0.1, 0.3, 1.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 1.0, 0.2],
        [0.0, 0.0, 0.0, 0.2, 1.0],
    ]
)
mu_B = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
Sigma_B = np.array(
    [
        [1.5, -0.7, 0.2, 0.0, 0.0],
        [-0.7, 1.5, 0.4, 0.0, 0.0],
        [0.2, 0.4, 1.5, 0.6, 0.0],
        [0.0, 0.0, 0.6, 1.5, 0.3],
        [0.0, 0.0, 0.0, 0.3, 1.5],
    ]
)

class_A = rng.multivariate_normal(mu_A, Sigma_A, size=N)
class_B = rng.multivariate_normal(mu_B, Sigma_B, size=N)
X1 = np.vstack([class_A, class_B])
y1 = np.array([0] * N + [1] * N)

# --- Dataset II: concentric shells ---
def sample_shell(n: int, radius_mean: float, radius_std: float, rng: np.random.Generator) -> np.ndarray:
    directions = rng.standard_normal((n, 5))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.normal(radius_mean, radius_std, size=n)
    return directions * radii[:, None]


class_C = sample_shell(N, 2.0, 0.4, rng)  # core
class_D = sample_shell(N, 5.0, 0.4, rng)  # shell
X2 = np.vstack([class_C, class_D])
y2 = np.array([0] * N + [1] * N)

# --- Figure 4: PCA projections side by side ---
pca1 = PCA(n_components=2).fit(X1)
X1_pca = pca1.transform(X1)
pca2 = PCA(n_components=2).fit(X2)
X2_pca = pca2.transform(X2)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for cls, name, color in [(0, "Class A", "#4C72B0"), (1, "Class B", "#DD8452")]:
    mask = y1 == cls
    axes[0].scatter(X1_pca[mask, 0], X1_pca[mask, 1], s=15, alpha=0.6, color=color, label=name)
axes[0].set_title("Dataset I (Gaussians) - PCA projection")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].legend()

for cls, name, color in [(0, "Class C (core)", "#55A868"), (1, "Class D (shell)", "#C44E52")]:
    mask = y2 == cls
    axes[1].scatter(X2_pca[mask, 0], X2_pca[mask, 1], s=15, alpha=0.6, color=color, label=name)
axes[1].set_title("Dataset II (concentric shells) - PCA projection")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].legend()

fig.suptitle("Figure 4 - PCA projections of Dataset I and Dataset II")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_pca_projections.png", dpi=150)
plt.close(fig)

ev1 = pca1.explained_variance_ratio_[:2].sum()
ev2 = pca2.explained_variance_ratio_[:2].sum()
print(f"Explained variance (PC1+PC2) - Dataset I: {ev1:.4f}")
print(f"Explained variance (PC1+PC2) - Dataset II: {ev2:.4f}")

# --- 5D center distances ---
dist1 = np.linalg.norm(class_A.mean(axis=0) - class_B.mean(axis=0))
dist2 = np.linalg.norm(class_C.mean(axis=0) - class_D.mean(axis=0))
print(f"5D distance between class centers - Dataset I: {dist1:.4f}")
print(f"5D distance between class centers - Dataset II: {dist2:.4f}")

# --- Figure 5: radius histograms ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(np.linalg.norm(class_A, axis=1), bins=30, alpha=0.6, color="#4C72B0", label="Class A")
axes[0].hist(np.linalg.norm(class_B, axis=1), bins=30, alpha=0.6, color="#DD8452", label="Class B")
axes[0].set_title("Dataset I - ||x|| distribution")
axes[0].set_xlabel("||x||")
axes[0].set_ylabel("Count")
axes[0].legend()

axes[1].hist(np.linalg.norm(class_C, axis=1), bins=30, alpha=0.6, color="#55A868", label="Class C (core)")
axes[1].hist(np.linalg.norm(class_D, axis=1), bins=30, alpha=0.6, color="#C44E52", label="Class D (shell)")
axes[1].set_title("Dataset II - ||x|| distribution")
axes[1].set_xlabel("||x||")
axes[1].set_ylabel("Count")
axes[1].legend()

fig.suptitle("Figure 5 - Radius histograms by class")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_radius_histograms.png", dpi=150)
plt.close(fig)

if __name__ == "__main__":
    pass
