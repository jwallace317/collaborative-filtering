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

    # convert pandas data frame to numpy matrix
    ice_cream = ice_cream_df.to_numpy()

    # get user input for restricted boltzmann machine initialization parameters
    print('Initialize Restricted Boltzmann Machine')
    num_hidden_units = int(input('number of hidden nodes in Restricted Boltzmann Machine: '))
    converge_constant = int(input('convergence constant for search and converge learning (enter 0 if constant learning is desired): '))

    # train the restricted boltzmann machine with varying learning rates
    learning_rates = [0.001, 0.01, 0.1]
    num_epochs = range(150)
    abs_mean_errors = np.zeros((len(learning_rates), len(num_epochs)))
    for i, learning_rate in enumerate(learning_rates):

        for j, epochs in enumerate(num_epochs):

            # initialize restricted boltzmann machine
            rbm = RestrictedBoltzmannMachine(num_visible_units=10, num_hidden_units=num_hidden_units)

            # train the restricted boltzmann machine
            rbm.train(ice_cream, learning_rate=learning_rate, max_num_epochs=epochs, converge_constant=converge_constant)

            # calculate the predictions
            predictions = rbm.predict_batch(ice_cream)

            # calculate absolute mean error
            abs_mean_error = rbm.absolute_mean_error(ice_cream, predictions)

            # append the error to the list
            abs_mean_errors[i, j] = abs_mean_error

    # plot the absolute mean error results
    plt.plot(num_epochs, abs_mean_errors[0, :], color='red', label=f'learning rate = { learning_rates[0] }')
    plt.plot(num_epochs, abs_mean_errors[1, :], color='green', label=f'learning rate = { learning_rates[1] }')
    plt.plot(num_epochs, abs_mean_errors[2, :], color='blue',  label=f'learning rate = { learning_rates[2] }')
    plt.title('Absolute Mean Error of Predicted Labels After Training')
    plt.xlabel('number of epochs')
    plt.ylabel('absolute mean error')
    plt.legend()
    plt.text(90, 4, f'number of hidden units = { num_hidden_units }')
    plt.text(90, 3.8, f'search and converge constant = { converge_constant }')
    plt.show()


if __name__ == '__main__':
    main()
