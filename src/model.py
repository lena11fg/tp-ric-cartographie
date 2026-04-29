import numpy as np


# ─────────────────────────────────────────────
# Fonctions d'activation
# ─────────────────────────────────────────────

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Dérivée de ReLU par rapport à z (avant activation)."""
    return (z > 0).astype(float)


def linear(z: np.ndarray) -> np.ndarray:
    return z


def linear_derivative(z: np.ndarray) -> np.ndarray:
    return np.ones_like(z)


# ─────────────────────────────────────────────
# MLP
# ─────────────────────────────────────────────

class MLP:
    """
    Paramètres
    ----------
    layer_sizes : liste d'entiers, ex. [2, 64, 64, 1]
    learning_rate : float
    momentum : float  (0 = SGD pur, ex. 0.9 = SGD+Momentum)
    """

    def __init__(self, layer_sizes: list[int],
                 learning_rate: float = 1e-3,
                 momentum: float = 0.9):

        assert len(layer_sizes) >= 2, "Il faut au moins une couche d'entrée et une de sortie."
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.momentum = momentum
        self.n_layers = len(layer_sizes) - 1   # nombre de couches de poids

        # Paramètres
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []

        # Vitesses pour le momentum
        self.vW: list[np.ndarray] = []
        self.vb: list[np.ndarray] = []

        self._init_weights()

        # Cache de la passe avant (nécessaire pour backprop)
        self._cache: dict = {}

    # ── Initialisation ──────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        rng = np.random.default_rng(0)
        for i in range(self.n_layers):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            is_output = (i == self.n_layers - 1)

            if is_output:
                # Xavier (sortie linéaire)
                std = np.sqrt(2.0 / (n_in + n_out))
            else:
                # He (couches ReLU)
                std = np.sqrt(2.0 / n_in)

            self.W.append(rng.normal(0, std, (n_in, n_out)))
            self.b.append(np.zeros((1, n_out)))
            self.vW.append(np.zeros((n_in, n_out)))
            self.vb.append(np.zeros((1, n_out)))

    # ── Passe avant (forward) ────────────────────────────────────────────────

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Propagation avant.
        X : (batch, n_in)
        Retourne : (batch, n_out)
        """
        A = X
        Zs, As = [], [X]

        for i in range(self.n_layers):
            Z = A @ self.W[i] + self.b[i]          # (batch, n_out_i)
            Zs.append(Z)

            if i < self.n_layers - 1:
                A = relu(Z)
            else:
                A = linear(Z)                       # couche de sortie : linéaire
            As.append(A)

        self._cache = {"Zs": Zs, "As": As}
        return A                                    # prédiction finale

    # ── Loss MSE ────────────────────────────────────────────────────────────

    @staticmethod
    def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    @staticmethod
    def mse_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """∂MSE/∂y_pred = 2*(y_pred - y_true)/N"""
        n = y_pred.shape[0]
        return 2.0 * (y_pred - y_true) / n

    # ── Rétropropagation (backprop) ─────────────────────────────────────────

    def backward(self, y_true: np.ndarray) -> dict[str, list]:
        """
        Calcul vectorisé des gradients par rétropropagation.
        Retourne un dict avec les listes dW et db.
        """
        Zs = self._cache["Zs"]
        As = self._cache["As"]

        dW = [None] * self.n_layers
        db = [None] * self.n_layers

        # Gradient de la perte par rapport à l'activation de sortie
        dA = self.mse_derivative(As[-1], y_true)   # (batch, n_out)

        # Rétropropagation couche par couche (de la sortie vers l'entrée)
        for i in reversed(range(self.n_layers)):
            Z = Zs[i]
            A_prev = As[i]                          # activation de la couche précédente

            if i == self.n_layers - 1:
                # Couche de sortie : activation linéaire → dérivée = 1
                dZ = dA * linear_derivative(Z)
            else:
                # Couches cachées : ReLU
                dZ = dA * relu_derivative(Z)

            # Gradients des paramètres de la couche i
            dW[i] = A_prev.T @ dZ                  # (n_in_i, n_out_i)
            db[i] = dZ.sum(axis=0, keepdims=True)  # (1, n_out_i)

            # Gradient à propager vers la couche précédente
            dA = dZ @ self.W[i].T                  # (batch, n_in_i)

        return {"dW": dW, "db": db}

    # ── Mise à jour des poids (SGD + Momentum) ──────────────────────────────

    def update(self, grads: dict) -> None:
        for i in range(self.n_layers):
            # v = β·v + (1-β)·grad  (formulation standard)
            self.vW[i] = self.momentum * self.vW[i] + (1 - self.momentum) * grads["dW"][i]
            self.vb[i] = self.momentum * self.vb[i] + (1 - self.momentum) * grads["db"][i]

            self.W[i] -= self.lr * self.vW[i]
            self.b[i] -= self.lr * self.vb[i]

    # ── Interface haut niveau ────────────────────────────────────────────────

    def train_step(self, X_batch: np.ndarray, y_batch: np.ndarray) -> float:
        """Un pas complet : forward → loss → backward → update."""
        y_pred = self.forward(X_batch)
        loss = self.mse(y_pred, y_batch)
        grads = self.backward(y_batch)
        self.update(grads)
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Passe avant sans mise à jour du cache de gradient."""
        A = X
        for i in range(self.n_layers):
            Z = A @ self.W[i] + self.b[i]
            A = relu(Z) if i < self.n_layers - 1 else linear(Z)
        return A