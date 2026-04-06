from data.dataset import generate_dataset, normalize
import matplotlib.pyplot as plt
import numpy as np

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
Zg = np.sin(np.sqrt(Xg**2 + Yg**2)) + 0.5 * np.cos(2*Xg + 2*Yg)

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