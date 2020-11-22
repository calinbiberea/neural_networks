import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import pickle
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from sklearn.metrics import mean_squared_error


class NeuralNetwork(nn.Module):
    """
        The neural network class
    """

    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        # Input to hidden first size
        first_hidden_layer_size = int((2 / 3) * input_size)
        self.first_hidden_layer = nn.Linear(input_size, first_hidden_layer_size)

        # First hidden layer to second hidden layer size
        second_hidden_layer_size = int((2 / 3) * first_hidden_layer_size)
        self.second_hidden_layer = nn.Linear(first_hidden_layer_size, second_hidden_layer_size)

        # Second hidden layer to output layer
        self.output_layer = nn.Linear(second_hidden_layer_size, output_size)

    def forward(self, x):
        x = torch.tanh(self.first_hidden_layer(x))
        x = torch.tanh(self.second_hidden_layer(x))
        x = self.output_layer(x)
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
        one_hot_encoded_x = pd.concat(
            [x.loc[:, x.columns != "ocean_proximity"], one_hot_encoded_pd_dataframe], axis=1)

        # Fill in any empty slots with 0s since now we operate with numbers
        one_hot_encoded_x.fillna(0, inplace=True)

        # Normalise the integers for better results and save if training
        if training:
            # Note that in training we do have the targets, so use those as well
            min_max_scaler = MinMaxScaler()

            # Fit the data on the scaler
            min_max_scaler.fit(one_hot_encoded_x)

            # Save the min_max_scaler as a parameter to be later used for normalising
            self.min_max_scaler = min_max_scaler

        # Retrieve the min_max_scaler and normalise
        min_max_scaler = self.min_max_scaler
        normalised_x = min_max_scaler.transform(one_hot_encoded_x)

        # Create the tensors
        tensor_x = torch.tensor(normalised_x, dtype=torch.float)
        tensor_y = torch.tensor(y.values, dtype=torch.float) if y is not None else None

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
        optimizer = optim.SGD(neural_network.parameters(), lr=0.01)

        # Use mean squared error for calculating the loss
        loss_function = nn.MSELoss()

        # Assume we have batches of size 50
        batch_size = 50

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
            return self.neural_network(preprocessed_x).numpy()

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

        preprocessed_x, preprocessed_y = self._preprocessor(x, y, training=False)

        with torch.no_grad():
            prediction = self.neural_network(preprocessed_x)

            return mean_squared_error(prediction, preprocessed_y)

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


def RegressorHyperParameterSearch():
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

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


def example_main():
    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't over-fitting
    regressor = Regressor(x_train, nb_epoch=10)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
