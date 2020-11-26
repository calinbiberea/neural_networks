import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        x = self.output_layer(x)
        return x


class Regressor:

    def __init__(self, x, nb_epoch=10, lr=0.08, batch_size=25):
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
        self.lr = lr
        self.batch_size = batch_size
        self.training_losses = []
        self.validation_losses = []

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

        if torch.cuda.is_available():
            tensor_x = tensor_x.cuda()
            tensor_y = tensor_y.cuda() if tensor_y is not None else None

        return tensor_x, tensor_y

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def fit(self, x, y, validation_data=None):
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

        if torch.cuda.is_available():
            neural_network.cuda()

        # Create optimizer
        optimizer = optim.SGD(neural_network.parameters(), lr=self.lr, momentum=0.8)

        # Use mean squared error for calculating the loss
        loss_function = nn.MSELoss()

        dataset = Data.TensorDataset(training_x, training_y)

        loader = Data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        training_losses = self.training_losses
        validation_losses = self.validation_losses

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

            training_losses.append(self.score(x, y))

            if validation_data:
                validation_x, validation_y = validation_data

                validation_losses.append(self.score(validation_x, validation_y))

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
            prediction = self.neural_network(preprocessed_x).cpu().numpy()

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

        prediction = self.predict(x)
        return np.sqrt(mean_squared_error(prediction, y))

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


# The most useful constant for cross_validation
NUMBER_OF_FOLDS = 10


def RegressorHyperParameterSearch(x_train_and_validation, y_train_and_validation,
                                  average_fold_size):
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
    # We do some form of cross-validation by keeping the best model through different training and validation
    x_folds = [x_train_and_validation.iloc[
               i * average_fold_size: i * average_fold_size + average_fold_size, :]
               for i in range(0, 9)]
    y_folds = [y_train_and_validation.iloc[
               i * average_fold_size: i * average_fold_size + average_fold_size, :]
               for i in range(0, 9)]

    def tune_parameter(x_splits, y_splits, parameter_to_tune):
        # Factor and number of parameters to check for
        if parameter_to_tune == "nb_epoch":
            PARAMETER_INCREASE = 50
            PARAMETER_COUNTS = 10
        elif parameter_to_tune == "lr":
            PARAMETER_INCREASE = 0.01
            PARAMETER_COUNTS = 10
        elif parameter_to_tune == "nb_batches":
            PARAMETER_INCREASE = 25
            PARAMETER_COUNTS = 5
        else:
            return

        parameter_choices = [PARAMETER_INCREASE * i for i in range(1, PARAMETER_COUNTS + 1)]
        parameter_rmse_scores = np.zeros(PARAMETER_COUNTS)

        for i in range(NUMBER_OF_FOLDS - 1):
            training_x = pd.concat([x_splits[k] for k in range(NUMBER_OF_FOLDS - 1) if k != i])
            training_y = pd.concat([y_splits[k] for k in range(NUMBER_OF_FOLDS - 1) if k != i])
            validation_x = x_splits[i]
            validation_y = y_splits[i]

            for parameter_index in range(PARAMETER_COUNTS):
                parameter_value = parameter_choices[parameter_index]
                network = None
                if parameter_to_tune == "nb_epoch":
                    network = Regressor(training_x, nb_epoch=parameter_value)
                elif parameter_to_tune == "lr":
                    network = Regressor(training_x, lr=parameter_value)
                elif parameter_to_tune == "nb_batches":
                    network = Regressor(training_x, batch_size=parameter_value)

                network.fit(training_x, training_y)

                rmse = network.score(validation_x, validation_y)
                parameter_rmse_scores[parameter_index] += rmse

        parameter_rmse_scores /= NUMBER_OF_FOLDS

        optimised_parameter = (parameter_rmse_scores.argmin() + 1) * PARAMETER_INCREASE
        print("After hyperparameter tuning for the parameter ", parameter_to_tune,
              " the optimal value is: ",
              optimised_parameter)

        return optimised_parameter

    # Parameters to tune
    learning_rate = tune_parameter(x_folds, y_folds, "lr")
    nb_epoch = tune_parameter(x_folds, y_folds, "nb_epoch")
    batch_size = tune_parameter(x_folds, y_folds, "batch_size")

    # Return the chosen hyper parameters
    return learning_rate, nb_epoch, batch_size


#######################################################################
#                       ** END OF YOUR CODE **
#######################################################################


def untuned_main(x_train_and_validation, y_train_and_validation, x_test, y_test, average_fold_size,
                 plot=False):
    # Training
    # You probably want to separate some held-out data
    # to make sure the model isn't over-fitting
    nb_epoch = 50

    TRAINING_SIZE = 8 * average_fold_size
    x_train = x_train_and_validation.iloc[:TRAINING_SIZE, :]
    x_validation = x_train_and_validation.iloc[TRAINING_SIZE:, :]
    y_train = y_train_and_validation.iloc[:TRAINING_SIZE, :]
    y_validation = y_train_and_validation.iloc[TRAINING_SIZE:, :]
    regressor = Regressor(x_train, nb_epoch)
    regressor.fit(x_train, y_train, validation_data=(x_validation, y_validation))

    if plot:
        epochs = range(1, nb_epoch + 1)

        train_line, = plt.plot(epochs, regressor.training_losses, label="train")
        validation_line, = plt.plot(epochs, regressor.validation_losses, label="validation")

        plt.legend(handles=[train_line, validation_line])

        plt.xlabel("epoch")
        plt.ylabel("rmse")
        plt.show()

    # No need to save the regressor since we tune right after, this is just for reference
    # save_regressor(regressor)

    # Error on testing data
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))


def tuned_main(x_train_and_validation, y_train_and_validation, x_test, y_test, average_fold_size):
    # Tuning
    lr, nb_epoch, batch_size = RegressorHyperParameterSearch(x_train_and_validation,
                                                             y_train_and_validation,
                                                             average_fold_size)

    # Train on validation data as well since it is more useful
    regressor = Regressor(x_train_and_validation, nb_epoch=1000, lr=0.08, batch_size=25)
    regressor.fit(x_train_and_validation, y_train_and_validation)
    save_regressor(regressor)

    # Error on testing data
    error = regressor.score(x_test, y_test)
    print("\nRegressor error: {}\n".format(error))

    # Save the regressor since this is probably the best we can generate
    save_regressor(regressor)


def example_main():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Shuffle the data
    shuffled_data = data.sample(frac=1)

    # Split data into training + validation and testing
    NUMBER_OF_DATA_POINTS = len(shuffled_data)
    AVERAGE_FOLD_SIZE = int(NUMBER_OF_DATA_POINTS / NUMBER_OF_FOLDS)
    TRAINING_AND_VALIDATION_SIZE = 9 * AVERAGE_FOLD_SIZE
    # Split
    data_train_and_validation = shuffled_data.iloc[:TRAINING_AND_VALIDATION_SIZE, :]
    data_test = shuffled_data.iloc[TRAINING_AND_VALIDATION_SIZE:, :]

    x_train_and_validation = data_train_and_validation.loc[:, data_test.columns != output_label]
    y_train_and_validation = data_train_and_validation.loc[:, [output_label]]
    x_test = data_test.loc[:, data_test.columns != output_label]
    y_test = data_test.loc[:, [output_label]]

    # In case you want to run an untuned regressor
    # untuned_main(x_train_and_validation, y_train_and_validation, x_test, y_test, AVERAGE_FOLD_SIZE)

    # Hyperparameter tuning and saving to pickle
    tuned_main(x_train_and_validation, y_train_and_validation, x_test, y_test, AVERAGE_FOLD_SIZE)


if __name__ == "__main__":
    example_main()
