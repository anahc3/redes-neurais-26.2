"""Exercise 1 - Point Clouds: geometry and spread in 2D."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

CLASSES = {
    0: {"mu": np.array([2.0, 3.0]), "sigma": np.array([0.8, 2.5])},
    1: {"mu": np.array([5.0, 6.0]), "sigma": np.array([1.2, 1.9])},
    2: {"mu": np.array([8.0, 1.0]), "sigma": np.array([0.9, 0.9])},
    3: {"mu": np.array([15.0, 4.0]), "sigma": np.array([0.5, 2.0])},
}
N_PER_CLASS = 100
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def generate(scale: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample all four classes with std scaled by `scale`; means are fixed."""
    points, labels = [], []
    for cls, params in CLASSES.items():
        pts = rng.normal(
            loc=params["mu"], scale=params["sigma"] * scale, size=(N_PER_CLASS, 2)
        )
        points.append(pts)
        labels.append(np.full(N_PER_CLASS, cls))
    return np.vstack(points), np.concatenate(labels)


def separation_ratios(scale: float) -> dict[tuple[int, int], float]:
    """r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j) at a given scale."""
    ratios = {}
    for i in range(4):
        for j in range(i + 1, 4):
            mu_i, mu_j = CLASSES[i]["mu"], CLASSES[j]["mu"]
            sbar_i = CLASSES[i]["sigma"].mean() * scale
            sbar_j = CLASSES[j]["sigma"].mean() * scale
            ratios[(i, j)] = np.linalg.norm(mu_i - mu_j) / (sbar_i + sbar_j)
    return ratios


def mixing_rate(points: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of points whose nearest class center is not their own class."""
    centers = np.array([CLASSES[c]["mu"] for c in range(4)])
    dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
    nearest = dists.argmin(axis=1)
    return float((nearest != labels).mean())


# --- Figure 1: base scatter (s=1) with class centers ---
X1, y1 = generate(1.0, rng)
plt.figure(figsize=(8, 6))
for cls in range(4):
    mask = y1 == cls
    plt.scatter(X1[mask, 0], X1[mask, 1], s=20, alpha=0.6, color=COLORS[cls], label=f"Class {cls}")
    plt.scatter(*CLASSES[cls]["mu"], marker="X", s=200, color=COLORS[cls], edgecolor="black", linewidth=1.5)
plt.title("Figure 1 - 2D point clouds (s=1.0) with class centers")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig1_point_clouds.png", dpi=150)
plt.close()

# --- Figure 2: 4 subplots for s in {0.5, 1.0, 2.0, 4.0}, shared axis limits ---
scales = [0.5, 1.0, 2.0, 4.0]
datasets = {s: generate(s, rng) for s in scales}

all_pts = np.vstack([pts for pts, _ in datasets.values()])
xlim = (all_pts[:, 0].min() - 1, all_pts[:, 0].max() + 1)
ylim = (all_pts[:, 1].min() - 1, all_pts[:, 1].max() + 1)

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
for ax, s in zip(axes.ravel(), scales):
    pts, labels = datasets[s]
    for cls in range(4):
        mask = labels == cls
        ax.scatter(pts[mask, 0], pts[mask, 1], s=15, alpha=0.6, color=COLORS[cls], label=f"Class {cls}")
    ax.set_title(f"s = {s}")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend()
fig.suptitle("Figure 2 - Spread scaling across s in {0.5, 1.0, 2.0, 4.0}")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_spread_scaling.png", dpi=150)
plt.close(fig)

# --- Separation ratio table at s=1 and s=2 ---
ratios_s1 = separation_ratios(1.0)
ratios_s2 = separation_ratios(2.0)
smallest_pair = min(ratios_s1, key=ratios_s1.get)

print("Separation ratios r_ij (s=1.0):")
for pair, r in sorted(ratios_s1.items(), key=lambda kv: kv[1]):
    print(f"  classes {pair}: r = {r:.4f}")
print(f"Smallest r_ij at s=1.0: pair {smallest_pair} -> r = {ratios_s1[smallest_pair]:.4f}")
print()
print("Separation ratios r_ij (s=2.0):")
for pair, r in sorted(ratios_s2.items(), key=lambda kv: kv[1]):
    print(f"  classes {pair}: r = {r:.4f}")

# --- Mixing rate at each scale ---
mixing_rates = {}
for s in scales:
    pts, labels = datasets[s]
    mixing_rates[s] = mixing_rate(pts, labels)
    print(f"Mixing rate at s={s}: {mixing_rates[s]:.4f}")

# --- Figure 3: mixing rate vs scale ---
plt.figure(figsize=(7, 5))
plt.plot(scales, [mixing_rates[s] for s in scales], marker="o", color="#4C72B0", label="Mixing rate")
plt.title("Figure 3 - Mixing rate vs. spread scale factor")
plt.xlabel("Scale factor s")
plt.ylabel("Mixing rate")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "fig3_mixing_rate.png", dpi=150)
plt.close()

if __name__ == "__main__":
    pass
