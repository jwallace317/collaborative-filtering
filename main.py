"""
Task Main Module
"""

# import necessary modules
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

    # train the restricted boltzmann machine
    rbm.train(ice_cream, learning_rate=0.1, max_num_epochs=1000)

    learning_rates = [0.01, 0.05, 0.1, 0.25]
    max_epochs = [10, 100, 1000]

    abs_mean_errors = {}
    for learning_rate in learning_rates:
        for epochs in max_epochs:
            rbm.train(ice_cream, learning_rate=learning_rate, max_num_epochs=epochs)

            abs_error_sum = 0
            for feature in ice_cream:
                prediction = rbm.predict(feature)

                abs_error = 0.5 * np.absolute(feature - prediction)
                abs_error_sum += np.sum(abs_error)
            abs_mean_error = abs_error_sum / ice_cream.shape[0]
            abs_mean_errors[(learning_rate, epochs)] = abs_mean_error

            print(f'abs mean error for learning rate { learning_rate } max epochs { epochs } = { abs_mean_errors[(learning_rate, epochs)] }')


if __name__ == '__main__':
    main()
