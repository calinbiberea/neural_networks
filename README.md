# The Python 3 files (.py) that our repository contains:

- part1_nn_lib.py
- part2_house_value_regression.py

# Our best model for part 2 is uploaded as:
- part2_model.pickle
 
# This repository requires the following imports in order to run:

- numpy
- pickle
- torch
- pandas
- sklearn
- matplotlib

# part1_nn_lib.py instuctions to run

- open a Linux terminal from the working directory 
- "python3 part1_nn_lib.py"
- press 'ENTER' 

This will run the code for part 1, but it will not output anything, since nothing was requested to be output in part 1.

# part2_house_value_regression.py instructions to run
- open a Linux terminal from the working directory 
- "python3 part2_house_value_regression.py"
- press 'ENTER'

This will run the code for part 2 and will output the regressor error. It will also create a pickle file which contains our model.

This version of the code calls tuned_main in the example_main function, which performs hyperparameter search before running the regression

An interesting function to look at in part 2 is the untuned_main one, which given the flag plot = True, plots the test and validation 
losses through an incresing number of epochs.

# [Optional] Instructions to change the Neural Netwrok architecture
- go to the NeuralNetwork class in part2_house_value_regression.py
- add/remove linear layers as seen fit and choose the size of hidden layers
- choose the activations functions for each layer in the forward() method

- go to the Regressor class in part2_house_value_regression.py
- change the default values for the number of epochs, learning rate and batch size in the Regressor constructor

# [Optional] Instructions to change the Tuning options
- go to tune_parameter() function inside the RegressorHyperParameterSearch
- change the PARAMETER_INCREASE and PARAMETER_COUNTS for each of the following: number of epochs, learning rate, batch size

