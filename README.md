# Restricted Boltzmann Machines - Collaborative Filtering

## CSE 5526 Introduction to Neural Networks: Lab 3 - Collaborative Filtering

This repository contains the python3 code necessary to perform the experiment requirements described in the lab 3 write up. This repository will create differing Restricted Boltzmann Machines to perform collaborative filtering on the provided ice cream data set. The main script initializes a machine with the given input parameters, trains it, and tests its efficacy by means of calculating its absolute average error associated with the training set. To effectively demonstrate the training process of these models, a matplotlib graph is printed to the screen displaying the training results for ease of comprehension.

## Getting Started

First, you will need to install python3 dependencies with the following command.

    pip install -r requirements.txt

Next, you can run the main script with the following command.

    python main.py

The main function will prompt the user for input about model architecture. Once both user input prompts have been completed the program begins to train and test the constructed Restricted Boltzmann Machine. The program terminates after printing the matplotlib graph to the screen.
