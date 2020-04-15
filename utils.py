"""
Utils Module
"""

# import necessary modules
import numpy as np


def sample(feature, lower_bound=0, upper_bound=1):
    """
    Sample

    This method will return the sampled target vector given the feature and the
    uniform distribution.

    Args:
        feature (np.array): feature vector, elements in (0, 1)
        lower_bound (int): lower bound of uniform distribution to sample from
        upper_bound (int): upper bound of uniform distribution to sample from

    Returns:
        sample (np.array): the sampled target vector, elements in {-1, 1}
    """

    sample = np.sign(feature - np.random.uniform(lower_bound, upper_bound))

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
