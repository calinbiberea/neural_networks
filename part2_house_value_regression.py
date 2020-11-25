import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.metrics import mean_squared_error


class NeuralNetwork(nn.Module):
    """
        The neural network class
    """

    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        # Input to hidden
        hidden_layer_size = int((3 / 2) * input_size)
        self.first_hidden_layer = nn.Linear(input_size, hidden_layer_size)
        self.second_hidden_layer = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.third_hidden_layer = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.fourth_hidden_layer = nn.Linear(hidden_layer_size, hidden_layer_size)

        # Second hidden layer to output layer
        self.output_layer = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        x = torch.relu(self.first_hidden_layer(x))
        x = torch.relu(self.second_hidden_layer(x))
        x = torch.relu(self.third_hidden_layer(x))
        x = torch.relu(self.fourth_hidden_layer(x))
        x = torch.sigmoid(self.output_layer(x))
        return x


class Regressor:

    def __init__(self, x, nb_epoch=1000):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # First preprocess and set the neural network parameters
        preprocessed_x, _ = self._preprocessor(x, training=True)
        self.input_size = preprocessed_x.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch

        # Then construct the neural network itself
        self.neural_network = NeuralNetwork(self.input_size, self.output_size)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y=None, training=False):
        """ 
        Preprocess input of the network.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} -- Preprocessed input array of size
                (batch_size, input_size).
            - {torch.tensor} -- Preprocessed target array of size
                (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
        # Return preprocessed x and y, return None for y if it was None

        # Deal with saving preprocess data into training data for one hot encoding
        if training:
            label_binarizer = LabelBinarizer()

            # Fill possibly empty slots and get the encodings
            filled_ocean_proximities = x["ocean_proximity"].fillna("N/A")
            ocean_proximity_features = list(filled_ocean_proximities.drop_duplicates())

            # Make the label binarizer fit the current encodings
            label_binarizer.fit(ocean_proximity_features)

            # Save the label binarizer
            self.label_binarizer = label_binarizer

            # Save the names of the created labels in the order saved in the binarizer
            self.ocean_proximity_features = label_binarizer.classes_

        # Retrieve the label binarizer and transform the ocean proximity column
        label_binarizer = self.label_binarizer
        one_hot_encoded_ocean_proximity = label_binarizer.transform(x["ocean_proximity"])

        # Create a panda dataframe for the new one hot encoded vector
        one_hot_encoded_pd_dataframe = pd.DataFrame(one_hot_encoded_ocean_proximity,
                                                    columns=self.ocean_proximity_features)

        # Drop the obsolete ocean_proximity feature and introduce the one hot encoded vector
        x.append(one_hot_encoded_pd_dataframe)
        one_hot_encoded_x = pd.DataFrame(x.loc[:, x.columns != "ocean_proximity"])

        # Get the average of every column
        means = one_hot_encoded_x.mean(axis=0)

        # Fill in any empty slots with 0s since now we operate with numbers
        one_hot_encoded_x.fillna(means, inplace=True)

        # Normalise the integers for better results and save if training
        if training:
            # Scale features
            feature_scaler = MinMaxScaler()

            # Fit the scaler to the features
            feature_scaler.fit(one_hot_encoded_x)

            # Save the scaler for later normalisation
            self.feature_scaler = feature_scaler

            if y is not None:
                # Scale output
                output_scaler = MinMaxScaler()

                # Fit scaler to the output
                output_scaler.fit(y)

                # Save the scaler for later normalisation
                self.output_scaler = output_scaler

        # Normalise features
        normalised_x = self.feature_scaler.transform(one_hot_encoded_x)

        normalised_y = self.output_scaler.transform(y) if y is not None else None

        # Create the tensors
        tensor_x = torch.tensor(normalised_x, dtype=torch.float)
        tensor_y = \
            torch.tensor(normalised_y, dtype=torch.float) if normalised_y is not None else None

        return tensor_x, tensor_y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Normalise the data and convert the pandas dataframe to a tensor
        training_x, training_y = self._preprocessor(x, y, training=True)

        nb_epoch = self.nb_epoch

        # Get the network as well
        neural_network = self.neural_network

        # Create optimizer
        optimizer = optim.SGD(neural_network.parameters(), lr=0.15, momentum=0.8)

        # Use mean squared error for calculating the loss
        loss_function = nn.MSELoss()

        # Assume we have batches of size 50
        batch_size = 25

        dataset = Data.TensorDataset(training_x, training_y)

        loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        # Train for given number of epochs
        for epoch in range(nb_epoch):
            # Execute Mini-batched Gradient Descent
            for batch_x, batch_y in loader:
                batch_x.requires_grad_(True)
                batch_y.requires_grad_(True)

                # Execute forward pass through the network
                prediction = neural_network(batch_x)

                # Compute the loss
                loss = loss_function(prediction, batch_y)

                # Zero the gradient buffers
                optimizer.zero_grad()

                # Do gradient descent
                loss.backward()

                # Update parameters
                optimizer.step()

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        preprocessed_x, _ = self._preprocessor(x, training=False)

        with torch.no_grad():
            prediction = self.neural_network(preprocessed_x).numpy()

            return self.output_scaler.inverse_transform(prediction)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        preprocessed_x, _ = self._preprocessor(x, y, training=False)

        with torch.no_grad():
            prediction = self.neural_network(preprocessed_x)

            rescaled_prediction = self.output_scaler.inverse_transform(prediction)

            return np.sqrt(mean_squared_error(rescaled_prediction, y))

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model):
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(x_train_val, y_train_val):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    # Tuning comes with the need for cross-validation, so do that
    NUMBER_OF_FOLDS = 10
    NUMBER_OF_DATA_POINTS = len(x_train_val)
    FOLD_SIZE = int(NUMBER_OF_DATA_POINTS / NUMBER_OF_FOLDS)
    splits_x = [x_train_val.loc[i: i + FOLD_SIZE - 1, :] for i in range(0, NUMBER_OF_DATA_POINTS, FOLD_SIZE)]
    splits_y = [y_train_val.loc[i: i + FOLD_SIZE - 1, :] for i in range(0, NUMBER_OF_DATA_POINTS, FOLD_SIZE)]

    # Given the size of our elements, remove one element from the dataset to not have obscure validation
    del splits_x[-1]
    del splits_y[-1]
    NUMBER_OF_FOLDS = NUMBER_OF_FOLDS - 1

    # The first hyperparameter that is worth looking at is the number of epochs
    # Pick some epochs in an upper bounded range (which we find by testing)
    EPOCH_TRIES = 20
    EPOCH_COUNT_JUMP = 10
    nb_epochs_choices = [EPOCH_COUNT_JUMP * i for i in range(1, EPOCH_TRIES + 1)]
    nb_epochs_choices_scores = np.zeros(EPOCH_TRIES)

    for i in range(NUMBER_OF_FOLDS):
        training_x = pd.concat([splits_x[k] for k in range(NUMBER_OF_FOLDS) if k != i])
        training_y = pd.concat([splits_y[k] for k in range(NUMBER_OF_FOLDS) if k != i])
        validation_x = splits_x[i]
        validation_y = splits_y[i]

        for epoch_choice_index in range(EPOCH_TRIES):
            nb_epoch = nb_epochs_choices[epoch_choice_index]
            network = Regressor(training_x, nb_epoch=nb_epoch)
            network.fit(training_x, training_y)

            rmse = network.score(validation_x, validation_y)
            nb_epochs_choices_scores[epoch_choice_index] = nb_epochs_choices_scores[epoch_choice_index] + rmse

    nb_epochs_choices_scores = nb_epochs_choices_scores / NUMBER_OF_FOLDS
    best_nb_epoch = (nb_epochs_choices_scores.argmin() + 1) * EPOCH_COUNT_JUMP
    print("A decent number of epochs to run is: ")
    print(best_nb_epoch)

    return best_nb_epoch  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def untuned_main(x_train, y_train, x_test, y_test):
    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't over-fitting
    regressor = Regressor(x_train, nb_epoch=50)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error on testing data
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


def example_main():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Shuffle the data
    shuffled_data = data.sample(frac=1)

    # Split data into training + validation and testing
    training_validation_split = int(0.9 * len(shuffled_data))
    training_validation_data = data.loc[:training_validation_split]
    testing_data = data.loc[training_validation_split:]

    # Split data
    training_validation_x = training_validation_data.loc[:, training_validation_data.columns != output_label]
    training_validation_y = training_validation_data.loc[:, [output_label]]
    test_x = testing_data.loc[:, testing_data.columns != output_label]
    test_y = testing_data.loc[:, [output_label]]

    # In case you want to run an untuned regressor
    untuned_main(training_validation_x, training_validation_y, test_x, test_y)

    # nb_epoch = RegressorHyperParameterSearch(training_validation_x, training_validation_y)
    #
    # regressor = Regressor(training_validation_x, nb_epoch=nb_epoch)
    # regressor.fit(training_validation_x, training_validation_y)
    # save_regressor(regressor)
    #
    # # Error on testing data
    # error = regressor.score(test_x, test_y)
    # print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
