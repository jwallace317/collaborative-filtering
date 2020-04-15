"""
Utils Module
"""

# import necessary modules
import numpy as np


def sample(feature):
    """
    Sample

    This method will return the sampled target vector given the feature vector.

    Args:
        feature (np.array): feature vector, elements in (0, 1)

    Returns:
        sample (np.array): the sampled target vector, elements in {-1, 1}
    """

    sample = np.sign(feature - np.random.uniform(0, 1, (feature.shape[0], feature.shape[1])))

    return sample


def sigmoid(feature):
    """
    Sigmoid

    This method will return the sigmoid distribution of a given feature vector.

    Args:
        feature (np.array): feature vector, elements in {-1, 1}

    Returns:
        sigmoid_dist (np.array): sigmoid distribution, elements in (0, 1)
    """

    sigmoid_dist = 1 / (1 + np.exp(-feature))

    return sigmoid_dist
