"""Shared VAE code: data split, model, hand-written loss, training and evaluation."""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "data"                     # git-ignored: MNIST and model checkpoints
CKPT_DIR = DATA_DIR / "vae_checkpoints"
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SEED = 42
LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 20
BETA = 1.0
EVAL_SEED = 1234                             # noise for evaluation, separate from training noise


def load_mnist():
    """Return flattened images in [0, 1] and labels: train (50k), validation (10k), test (10k)."""
    train_raw = datasets.MNIST(DATA_DIR, train=True, download=True)
    test_raw = datasets.MNIST(DATA_DIR, train=False, download=True)
    x_all = train_raw.data.float().div(255).view(-1, 784)
    y_all = train_raw.targets
    x_test = test_raw.data.float().div(255).view(-1, 784)
    y_test = test_raw.targets

    train_split, val_split = random_split(
        TensorDataset(x_all, y_all), [50_000, 10_000], generator=torch.Generator().manual_seed(SEED)
    )
    return {
        "train": (x_all[train_split.indices], y_all[train_split.indices]),
        "val": (x_all[val_split.indices], y_all[val_split.indices]),
        "test": (x_test, y_test),
    }


class VAE(nn.Module):
    """Encoder 784-512-256 with mu and log-variance heads; decoder d-256-512-784."""

    def __init__(self, d=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.mu_head = nn.Linear(256, d)
        self.logvar_head = nn.Linear(256, d)
        self.decoder = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 784), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, generator=None):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar, generator)
        return self.decode(z), mu, logvar


def reparameterize(mu, logvar, generator=None):
    """z = mu + sigma * eps with eps ~ N(0, I). The noise is outside the graph, so gradients reach mu and sigma."""
    sigma = torch.exp(0.5 * logvar)
    eps = torch.randn(sigma.shape, generator=generator)
    return mu + sigma * eps


def kl_per_dimension(mu, logvar):
    """Closed-form KL( N(mu, sigma^2) || N(0, 1) ) for each image and latent dimension."""
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())


def vae_loss(x_hat, x, mu, logvar, beta=BETA):
    """Return (total, reconstruction, KL), each in nats per image: summed over dimensions, averaged over the batch."""
    batch = x.size(0)
    reconstruction = F.binary_cross_entropy(x_hat, x, reduction="sum") / batch
    kl = kl_per_dimension(mu, logvar).sum() / batch
    return reconstruction + beta * kl, reconstruction, kl


@torch.no_grad()
def evaluate(model, x):
    """Average reconstruction and KL (nats per image) over x, with fixed evaluation noise."""
    model.eval()
    generator = torch.Generator().manual_seed(EVAL_SEED)
    total_rec, total_kl = 0.0, 0.0
    for batch in x.split(1024):
        x_hat, mu, logvar = model(batch, generator)
        _, rec, kl = vae_loss(x_hat, batch, mu, logvar)
        total_rec += rec.item() * batch.size(0)
        total_kl += kl.item() * batch.size(0)
    return total_rec / len(x), total_kl / len(x)


def train_vae(d, data):
    """Train one VAE with the specified budget. Returns the model and its per-epoch history."""
    torch.manual_seed(SEED)                  # required before building each model
    model = VAE(d)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    x_train, x_val = data["train"][0], data["val"][0]
    loader = DataLoader(TensorDataset(x_train), batch_size=BATCH_SIZE, shuffle=True)

    history = {key: [] for key in ("train_rec", "train_kl", "val_rec", "val_kl")}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        sum_rec, sum_kl = 0.0, 0.0
        for (batch,) in loader:
            x_hat, mu, logvar = model(batch)
            loss, rec, kl = vae_loss(x_hat, batch, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_rec += rec.item() * batch.size(0)
            sum_kl += kl.item() * batch.size(0)

        history["train_rec"].append(sum_rec / len(x_train))   # running average over the epoch
        history["train_kl"].append(sum_kl / len(x_train))
        val_rec, val_kl = evaluate(model, x_val)              # measured after the epoch's updates
        history["val_rec"].append(val_rec)
        history["val_kl"].append(val_kl)
        print(f"d={d} epoch {epoch:2d} | train rec {history['train_rec'][-1]:7.2f} kl {history['train_kl'][-1]:6.2f}"
              f" | val rec {val_rec:7.2f} kl {val_kl:6.2f}")
    return model, history


def load_or_train(d, data):
    """Reuse this model's checkpoint when it exists; otherwise train it with the fixed seed and save it."""
    path = CKPT_DIR / f"vae_d{d}.pt"
    if path.exists():
        saved = torch.load(path)
        model = VAE(d)
        model.load_state_dict(saved["state"])
        model.eval()
        return model, saved["history"]
    model, history = train_vae(d, data)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "history": history}, path)
    model.eval()
    return model, history
