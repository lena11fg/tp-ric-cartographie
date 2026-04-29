import numpy as np
import matplotlib.pyplot as plt

from src.model import MLP


def train(model: MLP,
          X_train: np.ndarray,
          y_train: np.ndarray,
          epochs: int = 800,
          batch_size: int = 64,
          seed: int = 1,
          verbose: bool = True,
          log_every: int = 50) -> list[float]:
    """
    Entraîne le modèle avec mini-batches aléatoires.

    Paramètres
    ----------
    model      : instance de MLP
    X_train    : (N, 2) — données normalisées
    y_train    : (N, 1) — cibles normalisées
    epochs     : nombre d'époques
    batch_size : taille des mini-batches
    seed       : graine pour la reproductibilité
    verbose    : affiche la loss tous les log_every epochs
    log_every  : fréquence d'affichage

    Retourne
    --------
    history : liste des losses moyennes par époque
    """
    rng = np.random.default_rng(seed)
    n_samples = X_train.shape[0]
    history: list[float] = []

    for epoch in range(1, epochs + 1):
        # Mélange aléatoire vectorisé
        idx = rng.permutation(n_samples)
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]

        epoch_losses: list[float] = []

        # Mini-batches — pas de boucle for sur les exemples individuels
        for start in range(0, n_samples, batch_size):
            X_batch = X_shuffled[start: start + batch_size]
            y_batch = y_shuffled[start: start + batch_size]
            loss = model.train_step(X_batch, y_batch)
            epoch_losses.append(loss)

        mean_loss = float(np.mean(epoch_losses))
        history.append(mean_loss)

        if verbose and epoch % log_every == 0:
            print(f"  Époque {epoch:>4}/{epochs}  |  Loss MSE : {mean_loss:.6f}")

    return history


def plot_loss(history: list[float], save_path: str | None = None) -> None:
    """Trace la courbe de loss en fonction des époques."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history, color="#2563eb", linewidth=1.5)
    ax.set_xlabel("Époque")
    ax.set_ylabel("MSE (données normalisées)")
    ax.set_title("Courbe d'apprentissage")
    ax.grid(True, alpha=0.3)

    # Annotation du minimum
    best_epoch = int(np.argmin(history))
    best_loss = history[best_epoch]
    ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.6, label=f"Min à ep. {best_epoch}")
    ax.scatter([best_epoch], [best_loss], color="red", zorder=5)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[train] Courbe de loss sauvegardée → {save_path}")
    plt.show()
