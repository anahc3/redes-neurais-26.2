"""Exercise 1 - A VAE on MNIST with d = 16: training, test numbers and Figures 1-3."""
import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from vae import FIG_DIR, RESULTS_DIR, kl_per_dimension, evaluate, load_mnist, load_or_train

FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

data = load_mnist()
model, history = load_or_train(16, data)
x_test, y_test = data["test"]

# --- C: final test numbers. This is the first time the test set is used. ---
test_rec, test_kl = evaluate(model, x_test)
print(f"test reconstruction {test_rec:.2f} | test KL {test_kl:.2f} | negative ELBO {test_rec + test_kl:.2f}")

# --- Figure 1: both loss terms against the epoch ---
epochs = np.arange(1, len(history["train_rec"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, term, name in [(axes[0], "rec", "Reconstruction"), (axes[1], "kl", "KL")]:
    ax.plot(epochs, history[f"train_{term}"], marker="o", ms=3, label="Train")
    ax.plot(epochs, history[f"val_{term}"], marker="o", ms=3, label="Validation")
    ax.set_title(f"{name} vs. epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Nats per image")
    ax.set_xticks(epochs[1::2])
    ax.grid(alpha=0.3)
    ax.legend()
fig.suptitle("Figure 1 - VAE training curves (d = 16)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_training_curves.png", dpi=150)
plt.close(fig)

# --- Figure 2: first test image of each digit, reconstructed from mu ---
first_indices = [int((y_test == digit).nonzero()[0]) for digit in range(10)]
with torch.no_grad():
    originals = x_test[first_indices]
    reconstructions = model.decode(model.encode(originals)[0])
fig, axes = plt.subplots(2, 10, figsize=(11, 2.9))
for digit in range(10):
    for row, images in enumerate([originals, reconstructions]):
        ax = axes[row, digit]
        ax.imshow(images[digit].view(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1, digit].set_xlabel(str(digit))
axes[0, 0].set_ylabel("Original")
axes[1, 0].set_ylabel("From μ")
fig.supxlabel("Digit (first test image of each class)", fontsize=10)
fig.suptitle("Figure 2 - Test images and their reconstructions (d = 16)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_reconstructions.png", dpi=150)
plt.close(fig)

# --- Figure 3: 8 x 8 samples from the prior ---
torch.manual_seed(0)
z_prior = torch.randn(64, 16)
with torch.no_grad():
    samples = model.decode(z_prior)
grid = samples.view(8, 8, 28, 28).permute(0, 2, 1, 3).reshape(8 * 28, 8 * 28)
fig, ax = plt.subplots(figsize=(6.5, 6.8))
ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
ax.set_xticks(np.arange(8) * 28 + 13.5, np.arange(1, 9))
ax.set_yticks(np.arange(8) * 28 + 13.5, np.arange(1, 9))
ax.set_xlabel("Grid column")
ax.set_ylabel("Grid row")
ax.set_title("Figure 3 - Decoded prior samples, z ~ N(0, I) (d = 16)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_prior_samples.png", dpi=150)
plt.close(fig)

# --- E: numbers supporting the analysis ---
with torch.no_grad():
    mu_test, logvar_test = model.encode(x_test)
    decoded_mu = model.decode(mu_test)
    # A larger prior sample gives a stable estimate; Figure 3 keeps its own 64 draws.
    prior_generator = torch.Generator().manual_seed(0)
    z_many = torch.randn(10_000, 16, generator=prior_generator)
    decoded_prior = model.decode(z_many)

    # Where prior samples land: distance to the closest encoded test image, against the
    # same distance between encoded test images (excluding the image itself).
    prior_to_mu = torch.cdist(z_many, mu_test).min(dim=1).values
    mu_to_mu = torch.cdist(mu_test, mu_test)
    mu_to_mu.fill_diagonal_(float("inf"))
    mu_to_mu = mu_to_mu.min(dim=1).values

    # What the decoder outputs: share of pixels that are neither near black nor near white.
    gray = lambda images: ((images > 0.1) & (images < 0.9)).float().sum(dim=1).mean().item()

    kl_by_dim = kl_per_dimension(mu_test, logvar_test).mean(dim=0)

    # Repeat the distances on the active dimensions only: dimensions with almost no KL
    # are ignored by the decoder, so their prior noise says nothing about holes.
    active = kl_by_dim > 0.5
    prior_to_mu_active = torch.cdist(z_many[:, active], mu_test[:, active]).min(dim=1).values
    mu_to_mu_active = torch.cdist(mu_test[:, active], mu_test[:, active])
    mu_to_mu_active.fill_diagonal_(float("inf"))
    mu_to_mu_active = mu_to_mu_active.min(dim=1).values

    # Reconstruction cost of decoding mu, against the sampled z used in test_rec.
    rec_from_mu = torch.nn.functional.binary_cross_entropy(decoded_mu, x_test, reduction="sum").item() / len(x_test)

# Sanity check of the closed-form KL against a hand-written Monte Carlo estimate.
check_generator = torch.Generator().manual_seed(0)
mu_c, logvar_c = mu_test[:200], logvar_test[:200]
sigma_c = (0.5 * logvar_c).exp()
eps = torch.randn((20_000, *mu_c.shape), generator=check_generator)
log_ratio = 0.5 * (eps.pow(2) * -1 - logvar_c + (mu_c + sigma_c * eps).pow(2))
kl_monte_carlo = log_ratio.sum(dim=2).mean().item()
kl_closed_form = kl_per_dimension(mu_c, logvar_c).sum(dim=1).mean().item()

results = {
    "test_rec": test_rec,
    "test_kl": test_kl,
    "test_neg_elbo": test_rec + test_kl,
    "history": history,
    "kl_closed_form_200_images": kl_closed_form,
    "kl_monte_carlo_200_images": kl_monte_carlo,
    "median_distance_prior_to_nearest_mu": prior_to_mu.median().item(),
    "median_distance_mu_to_nearest_mu": mu_to_mu.median().item(),
    "median_distance_prior_to_nearest_mu_active": prior_to_mu_active.median().item(),
    "median_distance_mu_to_nearest_mu_active": mu_to_mu_active.median().item(),
    "test_rec_decoding_mu": rec_from_mu,
    "gray_pixels_original": gray(x_test),
    "gray_pixels_decoded_mu": gray(decoded_mu),
    "gray_pixels_decoded_prior": gray(decoded_prior),
    "kl_by_dimension_min": kl_by_dim.min().item(),
    "kl_by_dimension_max": kl_by_dim.max().item(),
    "dimensions_above_half_nat": int((kl_by_dim > 0.5).sum()),
}
(RESULTS_DIR / "ex1.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps({k: v for k, v in results.items() if k != "history"}, indent=2))
