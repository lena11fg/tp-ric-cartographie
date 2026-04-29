"""
dataset.py — Génération et normalisation du dataset pour la fonction mystère.
f(x, y) = sin(sqrt(x^2 + y^2)) + 0.5 * cos(2x + 2y)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def target_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fonction mystère cible à approximer."""
    return np.sin(np.sqrt(x**2 + y**2)) + 0.5 * np.cos(2 * x + 2 * y)


def generate_dataset(n_points: int = 2000, x_min: float = -5.0, x_max: float = 5.0,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    
    rng = np.random.default_rng(seed)
    xy = rng.uniform(x_min, x_max, size=(n_points, 2))
    z = target_function(xy[:, 0], xy[:, 1]).reshape(-1, 1)
    return xy, z


class Normalizer:
    """Normalisation min-max vers [-1, 1] (réversible)."""

    def __init__(self):
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "Normalizer":
        self.min_ = data.min(axis=0)
        self.max_ = data.max(axis=0)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return 2.0 * (data - self.min_) / (self.max_ - self.min_) - 1.0

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return (data + 1.0) / 2.0 * (self.max_ - self.min_) + self.min_


def visualize_ground_truth(resolution: int = 100, save_path: str | None = None) -> None:
    """
    Affiche la heatmap et le scatter 3D de la fonction cible
    sur la grille [-5, 5] × [-5, 5].
    """
    x_vals = np.linspace(-5, 5, resolution)
    y_vals = np.linspace(-5, 5, resolution)
    XX, YY = np.meshgrid(x_vals, y_vals)
    ZZ = target_function(XX, YY)

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("Vérité terrain — f(x, y) = sin(√(x²+y²)) + 0.5·cos(2x+2y)",
                 fontsize=13, fontweight="bold")

    # — Heatmap —
    ax1 = fig.add_subplot(1, 2, 1)
    hm = ax1.contourf(XX, YY, ZZ, levels=50, cmap="RdYlBu_r")
    fig.colorbar(hm, ax=ax1, label="z = f(x, y)")
    ax1.set_title("Heatmap")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # — Surface 3D —
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(XX, YY, ZZ, cmap="RdYlBu_r", alpha=0.85, linewidth=0)
    ax2.set_title("Surface 3D")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[dataset] Visualisation sauvegardée → {save_path}")
    plt.show()