"""
Restricted Boltzmann Machine (RBM) Module
"""

# import necessary modules
import numpy as np

from utils import sample, sigmoid


class RestrictedBoltzmannMachine():
    """
    Restricted Boltzmann Machine Class

    This class is used to create Restricted Boltzmann Machines of varying
    dimensionality. Once created, a Restricted Boltzmann Machine represents a
    unidirectional, fully-connected, bipartite graph with weights. After
    training the weights, a Restricted Boltzmann Machine can be used for
    collaborative filtering.
    """

    def __init__(self, num_visible_units=1, num_hidden_units=1):

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
            prediction (np.array): predicted target vector, elements in {-1, 1}
        """

        # forward pass
        hidden = sample(sigmoid(np.dot(self.weights.T, feature)))

        # backward pass
        prediction = sample(sigmoid(np.dot(self.weights, hidden)))

        return prediction

    def predict_batch(self, features):
        """
        Predict Batch

        This method will predict a batch of predicted targets given a batch of
        features.

        Args:
            features (np.array): features matrix, elements in {-1, 1}

        Returns:
            predictions (np.array): predicted targets matrix, elements in {-1, 1}
        """

        # forward pass
        hidden = sample(sigmoid(np.dot(self.weights.T, features.T)))

        # backward pass
        predictions = sample(sigmoid(np.dot(self.weights, hidden))).T

        return predictions

    def absolute_mean_error(self, features, predictions):
        """
        Absolute Mean Error

        This method wil return the aboslute mean error of the given predictions
        and features.

        Args:
            features (np.array): features matrix
            predictions (np.array): predictions matrix

        Returns:
            abs_mean_error (float): absolute mean error of the matrices
        """

        abs_mean_error = np.sum(0.5 * np.absolute(predictions - features)) / features.shape[0]

        return abs_mean_error

    def train(self, features, learning_rate=0.1, max_num_epochs=10, converge_constant=0):
        """
        Train

        This method will train the Restricted Boltzmann Machine and update its
        weights using constrastive divergence for a desired number of epochs.

        Args:
            features (np.array): features matrix, elements in {-1, 1}
            learning_rate (float): the learning rate of the weight update rule
            max_num_epochs (int): max number of epochs to train
            converge_constant (int): constant to be used with search then converge learning
        """

        for epoch in range(max_num_epochs):

            # first forward pass
            H0 = sample(sigmoid(np.dot(self.weights.T, features.T)))

            # backward pass
            V1 = sample(sigmoid(np.dot(self.weights, H0)))

            # second forward pass
            H1 = sample(sigmoid(np.dot(self.weights.T, V1)))

            # update the weights
            if converge_constant:
                self.weights += (learning_rate / (1 + (epoch / converge_constant))) * (np.dot(features.T, H0.T) - np.dot(V1, H1.T))
            else:
                self.weights += learning_rate * (np.dot(features.T, H0.T) - np.dot(V1, H1.T))
