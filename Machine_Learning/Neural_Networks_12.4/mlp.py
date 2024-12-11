import numpy as np
import pprint

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class MLP:
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        self.W1 = np.random.rand(input_size, hidden_size_1)
        self.W2 = np.random.rand(hidden_size_1, hidden_size_2)
        self.W3 = np.random.rand(hidden_size_2, output_size)

        self.b1 = np.zeros((1, hidden_size_1))
        self.b2 = np.zeros((1, hidden_size_2))
        self.b3 = np.zeros((1, output_size))

    def forward_propagation(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = sigmoid(self.z3)

        return self.a3

    def back_propagation(self, X, y):
        dY = self.a3 - y

        dZ3 = dY * (self.a3 * (1 - self.a3))
        dW3 = np.dot(self.a2.T, dZ3)
        db3 = dZ3

        dZ2 = np.dot(dZ3, self.W3.T) * (self.a2 * (1 - self.a2))
        dW2 = np.dot(self.a1.T, dZ2)
        db2 = dZ2

        dZ1 = np.dot(dZ2, self.W2.T) * (self.a1 * (1 - self.a1))
        dW1 = np.dot(X.T, dZ1)
        db1 = dZ1

        return dW1, db1, dW2, db2, dW3, db3

    def update_weights(self, dW1, db1, dW2, db2, dW3, db3, learning_rate):
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3

    def fit(self, X, y, learning_rate=0.1):
        for i in range(len(X)):
            X_sample, y_sample = X[i].reshape(1, -1), y[i].reshape(1, -1)
            self.forward_propagation(X_sample)
            weights_bias = self.back_propagation(X_sample, y_sample)
            self.update_weights(*weights_bias, learning_rate)



if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    mlp = MLP(2, 12, 6 ,1)

    learning_rate = 0.5
    epochs = 10000

    for epoch in range(epochs):
        mlp.fit(X,y,learning_rate)
        if epoch % 100 == 0:
            total_loss = 0
            current_num = 0
            for i in range(len(X)):
                X_sample, y_sample = X[i].reshape(1, -1), y[i].reshape(1, -1)
                output = mlp.forward_propagation(X_sample)
                loss = -(y_sample * np.log(output) + (1 - y_sample) * np.log(1 - output))
                current_num += (np.round(output) == y_sample)
                total_loss += loss
            print(f'Epoch {epoch}: Loss:{total_loss / len(X)}, Acc = {current_num / len(X)}')
    
