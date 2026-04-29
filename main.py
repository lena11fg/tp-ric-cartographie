import os
import numpy as np

from src.dataset import generate_dataset, Normalizer, visualize_ground_truth
from src.model import MLP
from src.train import train, plot_loss
from src.visualize import compare_prediction
import sys
sys.stdout.reconfigure(encoding='utf-8')
# ──────────────────────────────────────────────
# Configuration — 
# ──────────────────────────────────────────────
CONFIG = {
    "n_points"     : 2000,
    "layer_sizes"  : [2, 64, 64, 1],  #[2, 4, 1], 
    "learning_rate": 8e-3,             
    "momentum"     : 0.9, #0.0,
    "epochs"       : 1000,             
    "batch_size"   : 32,               
    "log_every"    : 100,
    "output_dir"   : "outputs",
}

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    out = CONFIG["output_dir"]

    # ── Étape 1 : Génération du dataset ────────────────────────────────────
    print("=" * 60)
    print(" Étape 1 : Génération du dataset")
    print("=" * 60)

    X_raw, z_raw = generate_dataset(n_points=CONFIG["n_points"])
    print(f"  Dataset généré : {X_raw.shape[0]} points, X ∈ [-5, 5]²")
    print(f"  z ∈ [{z_raw.min():.3f}, {z_raw.max():.3f}]")

    # Normalisation
    x_norm = Normalizer()
    z_norm = Normalizer()
    X_train = x_norm.fit_transform(X_raw)
    y_train = z_norm.fit_transform(z_raw)
    print(f"  Après normalisation : X ∈ [{X_train.min():.2f}, {X_train.max():.2f}]"
          f" | y ∈ [{y_train.min():.2f}, {y_train.max():.2f}]")

    # Visualisation de la vérité terrain
    print("  → Visualisation de la vérité terrain…")
    visualize_ground_truth(resolution=100,
                           save_path=os.path.join(out, "ground_truth.png"))

    # ── Étape 2 : Construction du réseau ───────────────────────────────────
    print("\n" + "=" * 60)
    print(" Étape 2 : Architecture du réseau")
    print("=" * 60)

    model = MLP(
        layer_sizes=CONFIG["layer_sizes"],
        learning_rate=CONFIG["learning_rate"],
        momentum=CONFIG["momentum"],
    )
    total_params = sum(w.size + b.size for w, b in zip(model.W, model.b))
    print(f"  Architecture : {CONFIG['layer_sizes']}")
    print(f"  Paramètres totaux : {total_params:,}")
    print(f"  Learning rate : {CONFIG['learning_rate']}")
    print(f"  Momentum      : {CONFIG['momentum']}")

    # ── Étape 3 + 4 : Entraînement ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" Étape 3 & 4 : Entraînement (backpropagation vectorisée)")
    print("=" * 60)

    history = train(
        model=model,
        X_train=X_train,
        y_train=y_train,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        verbose=True,
        log_every=CONFIG["log_every"],
    )

    print(f"\n  Loss initiale : {history[0]:.6f}")
    print(f"  Loss finale   : {history[-1]:.6f}")
    print(f"  Réduction     : ×{history[0] / history[-1]:.1f}")

    # Courbe de loss
    plot_loss(history, save_path=os.path.join(out, "loss_curve.png"))

    # ── Test final : comparaison sur la grille ─────────────────────────────
    print("\n" + "=" * 60)
    print(" Test final : MLP vs Fonction originale")
    print("=" * 60)

    compare_prediction(
        model=model,
        x_normalizer=x_norm,
        y_normalizer=z_norm,
        resolution=80,
        save_path=os.path.join(out, "comparison.png"),
    )

    print("\n TP terminé. Figures sauvegardées dans le dossier 'outputs/'.")


if __name__ == "__main__":
    main()