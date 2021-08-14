import numpy as np


class NeuralNetwork:
    def __init__(self, neurons_count):
        self.alpha = 0.01
        self.weights_0 = 2 * np.random.random((3, neurons_count)) - 1
        self.weights_1 = 2 * np.random.random((neurons_count, 1)) - 1

        self.layer_0 = np.zeros((4,3))
        self.layer_1 = self.sigmoid(np.dot(self.layer_0, self.weights_0))
        self.layer_output = self.sigmoid(np.dot(self.layer_1, self.weights_1))
        self.errors = []
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def derivate_sigmoid(self, x):
        fx = self.sigmoid(x)
        return fx * (1 - fx)

    def mse_loss(self, y_true, y_pred):
        return ((y_true - y_pred)).mean()

    def train(self, train_set, train_output):

        l2_error = train_output - self.layer_output
        np.random.seed(1)
        for i in range(60000):
            self.errors.append(np.mean(l2_error))

            l2_delta = l2_error * self.derivate_sigmoid(self.layer_output)
            l1_error = l2_delta.dot(self.weights_1.T)
            l1_delta = l1_error * self.derivate_sigmoid(self.layer_1)

            self.weights_1 += self.alpha * self.layer_1.T.dot(l2_delta)
            self.weights_0 += self.alpha * self.layer_0.T.dot(l1_delta)

        return self.weights_0, self.weights_1

    def get_neural_output(weights_0, weights_1, data):
        a = np.dot(data, weights_0)
        b = np.dot(a, weight_1)
        return sigmoid(b)

    def get_errors(self):
        return self.errors