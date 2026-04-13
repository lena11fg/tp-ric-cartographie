from model.layers import Dense
from model.activations import relu

class MLP:
    def __init__(self):
        # Architecture [2, 64, 64, 1]
        self.layer1 = Dense(2, 64)
        self.layer2 = Dense(64, 64)
        self.layer3 = Dense(64, 1)

    def forward(self, X):
        self.z1 = self.layer1.forward(X)
        self.a1 = relu(self.z1)

        self.z2 = self.layer2.forward(self.a1)
        self.a2 = relu(self.z2)

        self.z3 = self.layer3.forward(self.a2)  # sortie linéaire
        return self.z3