import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.dataset import target_function, Normalizer
from src.model import MLP


def compare_prediction(model: MLP,
                       x_normalizer: Normalizer,
                       y_normalizer: Normalizer,
                       resolution: int = 80,
                       save_path: str | None = None) -> None:
    """
    Génère une grille [-5, 5]² et compare :
      - la surface originale f(x, y)
      - la prédiction du MLP (dénormalisée)
    en heatmap et en surface 3D.
    """
    x_vals = np.linspace(-5, 5, resolution)
    y_vals = np.linspace(-5, 5, resolution)
    XX, YY = np.meshgrid(x_vals, y_vals)

    # Vérité terrain
    ZZ_true = target_function(XX, YY)

    # Prédiction du MLP
    XY_grid = np.column_stack([XX.ravel(), YY.ravel()])  # (R², 2)
    XY_norm = x_normalizer.transform(XY_grid)
    Z_pred_norm = model.predict(XY_norm)                  # (R², 1)
    Z_pred = y_normalizer.inverse_transform(Z_pred_norm).reshape(resolution, resolution)

    # Erreur absolue
    ZZ_err = np.abs(ZZ_true - Z_pred)

    # ── Figure principale : heatmaps côte à côte ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("MLP vs Fonction originale", fontsize=14, fontweight="bold")

    vmin = min(ZZ_true.min(), Z_pred.min())
    vmax = max(ZZ_true.max(), Z_pred.max())

    for ax, data, title in zip(
            axes,
            [ZZ_true, Z_pred, ZZ_err],
            ["Vérité terrain", "Prédiction MLP", "Erreur absolue"]):
        levels = 50
        if title == "Erreur absolue":
            cf = ax.contourf(XX, YY, data, levels=levels, cmap="hot_r")
        else:
            cf = ax.contourf(XX, YY, data, levels=levels, cmap="RdYlBu_r",
                             vmin=vmin, vmax=vmax)
        fig.colorbar(cf, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    if save_path:
        path_hm = save_path.replace(".png", "_heatmap.png")
        plt.savefig(path_hm, dpi=150, bbox_inches="tight")
        print(f"[visualize] Heatmap sauvegardée → {path_hm}")
    plt.show()

    # ── Surfaces 3D côte à côte ─────────────────────────────────────────────
    fig3d = plt.figure(figsize=(14, 5))
    fig3d.suptitle("Surfaces 3D — MLP vs Fonction originale",
                   fontsize=13, fontweight="bold")

    for idx, (data, title) in enumerate([(ZZ_true, "Vérité terrain"),
                                          (Z_pred, "Prédiction MLP")], 1):
        ax = fig3d.add_subplot(1, 2, idx, projection="3d")
        ax.plot_surface(XX, YY, data, cmap="RdYlBu_r", alpha=0.85, linewidth=0)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    plt.tight_layout()
    if save_path:
        path_3d = save_path.replace(".png", "_3d.png")
        plt.savefig(path_3d, dpi=150, bbox_inches="tight")
        print(f"[visualize] Surface 3D sauvegardée → {path_3d}")
    plt.show()

    # Métriques
    mse_test = float(np.mean((ZZ_true - Z_pred) ** 2))
    mae_test = float(np.mean(ZZ_err))
    print(f"\n[visualize] MSE test (domaine complet) : {mse_test:.6f}")
    print(f"[visualize] MAE test (domaine complet) : {mae_test:.6f}")