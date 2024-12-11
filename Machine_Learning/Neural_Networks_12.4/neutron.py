import numpy as np

class Neutron:
    def __init__(self):
        self.weights = [0.3, 0.9]
        self.threshold = 0.05

    def activate(self, z):
        return 0 if z < self.threshold else 1

    def predict(self, x):
        linear_output = np.dot(self.weights, x)
        predicted_y = self.activate(linear_output)
        return predicted_y


if __name__ == "__main__":
    neutron = Neutron()
    x = [-1, 1]
    print(neutron.predict(x))
