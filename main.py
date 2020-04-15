"""
Task Main Module
"""

# import necessary modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rbm import RestrictedBoltzmannMachine


# task main
def main():
    """
    Task Main
    """

    # read in ice cream csv data
    ice_cream_df = pd.read_csv('./icecream.csv', sep=',', header=None)
    print(ice_cream_df)

    # convert pandas data frame to numpy matrix
    ice_cream = ice_cream_df.to_numpy()
    print(ice_cream)

    # initialize restricted boltzmann machine with 10 visible nodes, 4 hidden nodes
    rbm = RestrictedBoltzmannMachine(10, 4)

    errors = []
    for epochs in range(250):
        rbm.train(ice_cream, learning_rate=0.01, max_num_epochs=epochs)

        predictions = rbm.predict_batch(ice_cream)

        abs_mean_error = np.sum(0.5 * np.absolute(predictions - ice_cream)) / ice_cream.shape[0]
        errors.append(abs_mean_error)

    plt.plot(range(250), errors)
    plt.show()


if __name__ == '__main__':
    main()
