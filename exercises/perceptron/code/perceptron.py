"""From-scratch single-layer Perceptron (step activation) with optional pocket tracking."""
from dataclasses import dataclass

import numpy as np


def step(z: np.ndarray) -> np.ndarray:
    """Heaviside step: 1 if z >= 0 else 0."""
    return (z >= 0).astype(int)


@dataclass
class FitResult:
    w: np.ndarray
    b: float
    epochs_ran: int
    converged: bool
    acc_history: list
    pocket_w: np.ndarray = None
    pocket_b: float = None
    pocket_acc: float = None
    pocket_epoch: int = None
    pocket_acc_history: list = None


class Perceptron:
    """yhat = step(w.x + b); update w += eta*(y-yhat)*x, b += eta*(y-yhat)."""

    def __init__(self, eta: float = 0.01, max_epochs: int = 100,
                 w_init: np.ndarray = None, b_init: float = 0.0):
        self.eta = eta
        self.max_epochs = max_epochs
        self.w_init = w_init
        self.b_init = b_init

    def fit(self, X: np.ndarray, y: np.ndarray, track_pocket: bool = False) -> FitResult:
        w = np.array(self.w_init, dtype=float) if self.w_init is not None else np.zeros(X.shape[1])
        b = float(self.b_init)

        acc_history = []
        pocket_acc_history = [] if track_pocket else None
        init_acc = float((step(X @ w + b) == y).mean())
        pocket_w, pocket_b, pocket_acc, pocket_epoch = w.copy(), b, init_acc, 0

        converged = False
        epoch = 0
        for epoch in range(1, self.max_epochs + 1):
            updated = False
            for xi, yi in zip(X, y):
                yhat = int(w @ xi + b >= 0)
                if yhat != yi:
                    w = w + self.eta * (yi - yhat) * xi
                    b = b + self.eta * (yi - yhat)
                    updated = True
                    if track_pocket:
                        acc = float((step(X @ w + b) == y).mean())
                        if acc > pocket_acc:
                            pocket_w, pocket_b, pocket_acc, pocket_epoch = w.copy(), b, acc, epoch
            acc = float((step(X @ w + b) == y).mean())
            acc_history.append(acc)
            if track_pocket:
                pocket_acc_history.append(pocket_acc)
            if not updated:
                converged = True
                break

        return FitResult(
            w=w, b=b, epochs_ran=epoch, converged=converged, acc_history=acc_history,
            pocket_w=pocket_w if track_pocket else None,
            pocket_b=pocket_b if track_pocket else None,
            pocket_acc=pocket_acc if track_pocket else None,
            pocket_epoch=pocket_epoch if track_pocket else None,
            pocket_acc_history=pocket_acc_history,
        )
