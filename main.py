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

    The main method of this program.
    """

    # read in ice cream csv data
    ice_cream_df = pd.read_csv('./icecream.csv', sep=',', header=None)
    print(ice_cream_df)

    # convert pandas data frame to numpy matrix
    ice_cream = ice_cream_df.to_numpy()
    print(ice_cream)

    # initialize restricted boltzmann machine
    rbm = RestrictedBoltzmannMachine(10, 4)

    # train the restricted boltzmann machine
    rbm.train(ice_cream, max_epochs=100)

    print(rbm.predict(ice_cream))

    return 0


if __name__ == '__main__':
    main()
