"""
Restricted Boltzmann Machine (RBM) Module
"""

# import necessary modules
import numpy as np
from sklearn.utils import shuffle

from utils import sample, sigmoid


class RestrictedBoltzmannMachine():
    """
    Restricted Boltzmann Machine Class

    This class is used to instantiate Restricted Boltzmann Machines. Once
    instantiated, these machines can be trained and used to predict feature
    data.
    """

    def __init__(self, num_visible_units, num_hidden_units):
        # random seed
        np.random.seed(1)

        # initialize the weights
        self.weights = np.random.rand(num_visible_units, num_hidden_units)

    def predict(self, feature):
        """
        Predict

        This method will predict a target vector given a feature vector.

        Args:
            feature (np.array): feature vector, elements in {-1, 1}

        Returns:
            prediction (np.array): the predicted target vector, elements in {-1, 1}
        """

        hidden = sample(sigmoid(np.dot(self.weights.T, feature)))
        prediction = sample(sigmoid(np.dot(self.weights, hidden)))

        return prediction

    def predict_batch(self, features):
        """
        Predict

        This method will predict a target matrix given a features matrix.

        Args:
            features (np.array): features matrix, elements in {-1, 1}

        Returns:
            predictions (np.array): the predicted target matrix, elements in {-1, 1}
        """

        hidden = sample(sigmoid(np.dot(self.weights.T, features.T)))
        predictions = sample(sigmoid(np.dot(self.weights, hidden)))

        return predictions.T

    def train(self, features, learning_rate=0.1, max_num_epochs=10):
        """
        Train

        This method will train the Restricted Boltzmann Machine weights using
        constrastive divergence for a certain number of epochs.

        Args:
            features (np.array): features matrix, elements in {-1, 1}
            max_num_epochs (int): max number of epochs to train
            learning_rate (float): the learning rate of the weight update rule
        """

        for epoch in range(max_num_epochs):
            H0 = sample(sigmoid(np.dot(self.weights.T, features.T)))
            V1 = sample(sigmoid(np.dot(self.weights, H0)))
            H1 = sample(sigmoid(np.dot(self.weights.T, V1)))

            self.weights += learning_rate * (np.dot(features.T, H0.T) - np.dot(V1, H1.T))

            # for row in range(self.weights.shape[0]):
            #     for col in range(self.weights.shape[1]):
            #         self.weights[row, col] = self.weights[row, col] + learning_rate * (np.mean(np.multiply(features[:, row], H0[col, :])) - np.mean(np.multiply(V1[row, :], H1[col, :])))
