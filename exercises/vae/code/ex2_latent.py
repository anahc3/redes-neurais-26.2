"""Exercise 2 - A 2-D latent space: test numbers, Figures 4-6 and analysis numbers."""
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors

from vae import FIG_DIR, RESULTS_DIR, evaluate, load_mnist, load_or_train

FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Figure 6 endpoints: first test images of digits 3 and 4 (test indices 18 and 4).
INDEX_A, INDEX_B = 18, 4

data = load_mnist()
model_2d, history_2d = load_or_train(2, data)
model_16d, _ = load_or_train(16, data)
x_test, y_test = data["test"]

# --- A: final test numbers, beside d = 16 ---
test_rec, test_kl = evaluate(model_2d, x_test)
test_rec_16, test_kl_16 = evaluate(model_16d, x_test)
print(f"d=2  test reconstruction {test_rec:.2f} | KL {test_kl:.2f}")
print(f"d=16 test reconstruction {test_rec_16:.2f} | KL {test_kl_16:.2f}")

# --- Figure 4: mu of all 10,000 test images ---
with torch.no_grad():
    mu_2d = model_2d.encode(x_test)[0]
mu_np, labels = mu_2d.numpy(), y_test.numpy()
fig, ax = plt.subplots(figsize=(8, 7))
cmap = plt.get_cmap("tab10")
for digit in range(10):
    mask = labels == digit
    ax.scatter(mu_np[mask, 0], mu_np[mask, 1], s=5, alpha=0.6, color=cmap(digit), label=str(digit))
ax.set_title("Figure 4 - Encoded test set: μ of 10,000 images (d = 2)")
ax.set_xlabel("μ₁")
ax.set_ylabel("μ₂")
ax.legend(title="Digit", markerscale=3, loc="center left", bbox_to_anchor=(1.01, 0.5))
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_latent_scatter.png", dpi=150)
plt.close(fig)

# --- Figure 5: decoded plane on the inverse-CDF grid ---
quantiles = np.linspace(0.05, 0.95, 15)
axis_z = norm.ppf(quantiles)
z1, z2 = np.meshgrid(axis_z, axis_z[::-1])          # top row = largest z2, like the axes of Figure 4
plane = torch.tensor(np.column_stack([z1.ravel(), z2.ravel()]), dtype=torch.float32)
with torch.no_grad():
    tiles = model_2d.decode(plane)
canvas = tiles.view(15, 15, 28, 28).permute(0, 2, 1, 3).reshape(15 * 28, 15 * 28)
fig, ax = plt.subplots(figsize=(9, 9.2))
ax.imshow(canvas, cmap="gray", vmin=0, vmax=1)
tick_positions = np.arange(15) * 28 + 13.5
ax.set_xticks(tick_positions, [f"{v + 0:.2f}".replace("-0.00", "0.00") for v in axis_z], rotation=90)
ax.set_yticks(tick_positions, [f"{v + 0:.2f}".replace("-0.00", "0.00") for v in axis_z[::-1]])
ax.set_xlabel("z₁ = Φ⁻¹(q)")
ax.set_ylabel("z₂ = Φ⁻¹(q)")
ax.set_title("Figure 5 - Decoded latent plane on the Φ⁻¹ grid (d = 2)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig5_decoded_plane.png", dpi=150)
plt.close(fig)

# --- Figure 6: interpolation in pixel space and in the d = 16 latent space ---
ts = torch.linspace(0, 1, 10)
x_a, x_b = x_test[INDEX_A], x_test[INDEX_B]
pixel_row = torch.stack([(1 - t) * x_a + t * x_b for t in ts])
with torch.no_grad():
    mu_a = model_16d.encode(x_a.unsqueeze(0))[0][0]
    mu_b = model_16d.encode(x_b.unsqueeze(0))[0][0]
    latent_row = model_16d.decode(torch.stack([(1 - t) * mu_a + t * mu_b for t in ts]))
fig, axes = plt.subplots(2, 10, figsize=(11, 2.9))
for column, t in enumerate(ts):
    for row, images in enumerate([pixel_row, latent_row]):
        ax = axes[row, column]
        ax.imshow(images[column].view(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1, column].set_xlabel(f"{t:.2f}")
axes[0, 0].set_ylabel("Pixel")
axes[1, 0].set_ylabel("Latent")
fig.supxlabel("Blend t", fontsize=10)
fig.suptitle(f"Figure 6 - Interpolation between test images {INDEX_A} (digit {int(y_test[INDEX_A])}) "
             f"and {INDEX_B} (digit {int(y_test[INDEX_B])})")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig6_interpolation.png", dpi=150)
plt.close(fig)

# --- E: numbers supporting the analysis ---
# Overlap in Figure 4: among each point's 10 nearest neighbors, the share from each other digit.
neighbors = NearestNeighbors(n_neighbors=11).fit(mu_np).kneighbors(mu_np, return_distance=False)[:, 1:]
mix = np.zeros((10, 10))
for digit in range(10):
    neighbor_labels = labels[neighbors[labels == digit]].ravel()
    mix[digit] = np.bincount(neighbor_labels, minlength=10) / len(neighbor_labels)
pairs = sorted(
    ((a, b, (mix[a, b] + mix[b, a]) / 2) for a in range(10) for b in range(a + 1, 10)),
    key=lambda pair: -pair[2],
)

# Variation kept within a digit: average per-pixel standard deviation among images of the same class.
with torch.no_grad():
    decoded_2d = model_2d.decode(mu_2d)
    decoded_16d = model_16d.decode(model_16d.encode(x_test)[0])
same_digit_spread = lambda images: float(np.mean([
    images[y_test == digit].std(dim=0).mean().item() for digit in range(10)
]))

results = {
    "d2_test_rec": test_rec,
    "d2_test_kl": test_kl,
    "d16_test_rec": test_rec_16,
    "d16_test_kl": test_kl_16,
    "d2_history": history_2d,
    "interpolation_indices": [INDEX_A, INDEX_B],
    "interpolation_digits": [int(y_test[INDEX_A]), int(y_test[INDEX_B])],
    "top_overlap_pairs": [{"a": a, "b": b, "neighbor_share": share} for a, b, share in pairs[:5]],
    "same_digit_pixel_std_original": same_digit_spread(x_test),
    "same_digit_pixel_std_decoded_d16": same_digit_spread(decoded_16d),
    "same_digit_pixel_std_decoded_d2": same_digit_spread(decoded_2d),
    "cell_prior_mass": float((quantiles[1] - quantiles[0]) ** 2),
}
(RESULTS_DIR / "ex2.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps({k: v for k, v in results.items() if k != "d2_history"}, indent=2))
