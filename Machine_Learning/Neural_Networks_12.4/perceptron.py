import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, num_epochs=100) -> None:
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weights: np.ndarray = None
        self.bias = 0

    def activate(self, z):
        return 1 if z >= 0 else 0

    def predict(self, x):
        z = np.dot(self.weights, x) + self.bias
        predicted_y = self.activate(z)
        return predicted_y

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)

        for epoch in range(self.num_epochs):
            for x_i, y_i in zip(X, y):
                predicted_y_i = self.predict(x_i)
                self.weights += self.learning_rate * (y_i - predicted_y_i) * x_i
                self.bias += self.learning_rate * (y_i - predicted_y_i)

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, 1, 1, -1])
    perceptron = Perceptron(num_epochs=1000)
    perceptron.fit(X, y)
    print(perceptron.weights,perceptron.bias)

    for x_i, y_i in zip(X, y):
        predicted_y_i = perceptron.predict(x_i)
        print(y_i, predicted_y_i)
