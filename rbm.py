"""
Restricted Boltzmann Machine Module
"""

# import necessary modules
import numpy as np

from utils import sample, sigmoid


class RestrictedBoltzmannMachine():

    def __init__(self, num_visible_units, num_hidden_units):
        # random seed
        np.random.seed(1)

        # initialize the weights
        self.weights = np.random.rand(num_visible_units, num_hidden_units)

    def predict(self, features):
        sum_errors = 0
        for feature in features:
            hidden0 = sample(sigmoid(np.dot(self.weights.T, feature)))
            prediction = sample(sigmoid(np.dot(self.weights, hidden0)))

            sum_errors += np.sum(np.absolute(feature - prediction))
        avg_abs_error = sum_errors / features.shape[0]

        return avg_abs_error

    def train(self, features, max_epochs=100, learning_rate=0.1):
        H0 = np.zeros((4, 120))
        V1 = np.zeros((10, 120))
        H1 = np.zeros((4, 120))

        for epoch in range(max_epochs):
            for i, feature in enumerate(features):
                hidden0 = sample(sigmoid(np.dot(self.weights.T, feature)))
                visible1 = sample(sigmoid(np.dot(self.weights, hidden0)))
                hidden1 = sample(sigmoid(np.dot(self.weights.T, visible1)))

                H0[:, i] = hidden0
                V1[:, i] = visible1
                H1[:, i] = hidden1

            for row in range(self.weights.shape[0]):
                for col in range(self.weights.shape[1]):
                    self.weights[row, col] = self.weights[row, col] + learning_rate * (np.mean(np.multiply(features[:, row], H0[col, :])) - np.mean(np.multiply(V1[row, :], H1[col, :])))
