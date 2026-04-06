import numpy as np

# Fonction f(x, y)
def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2)) + 0.5 * np.cos(2*x + 2*y)

# Générer dataset
def generate_dataset(n=2000):
    np.random.seed(42)
    X = np.random.uniform(-5, 5, (n, 2))
    
    x = X[:, 0]
    y = X[:, 1]
    
    z = f(x, y)
    
    return X, z

# Normalisation
def normalize(X, z):
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
    z_norm = (z - z.mean()) / z.std()
    
    return X_norm, z_norm