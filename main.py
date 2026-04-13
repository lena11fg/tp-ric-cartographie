from data.dataset import generate_dataset, normalize ,f
import matplotlib.pyplot as plt
import numpy as np
from model.mlp import MLP
from model.loss import mse
# Générer données
X, z = generate_dataset()

# Normaliser
X_norm, z_norm = normalize(X, z)

# Vérification
print("Shape X:", X.shape)
print("Shape z:", z.shape)

# données
x = X[:, 0]
y = X[:, 1]

# créer grille
grid_size = 100
x_lin = np.linspace(-5, 5, grid_size)
y_lin = np.linspace(-5, 5, grid_size)
Xg, Yg = np.meshgrid(x_lin, y_lin)
Zg = f(Xg, Yg)

# =========================
# FIGURE AVEC 2 PLOTS
# =========================
fig = plt.figure(figsize=(12, 5))

# ---- Scatter 3D ----
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(x, y, z)
ax1.set_title("Scatter 3D")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# ---- Heatmap ----
ax2 = fig.add_subplot(122)
im = ax2.imshow(Zg, extent=[-5, 5, -5, 5], origin='lower')
ax2.set_title("Heatmap")
plt.colorbar(im, ax=ax2)

plt.show()

# modèle
model = MLP()
# forward
y_pred = model.forward(X_norm)
# loss
loss_value = mse(y_pred, z_norm)
print("Loss:", loss_value)