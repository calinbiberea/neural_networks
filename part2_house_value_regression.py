import torch
import torch.nn as nn
import pickle
import pandas as pd
import random
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler


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
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch

        # Then construct the neural network itself

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

        # Transform x into numpy nd_array
        # First fill all the empty values, then binarize the 9th column of the dataset (ocean_proximity)
        if training:
            label_binarizer = LabelBinarizer()
            label_binarizer.y_type_ = "multiclass"

            x["ocean_proximity"].fillna("N/A")

            ocean_proximity_features = list(x["ocean_proximity"].drop_duplicates())

            print("New features: " + str(ocean_proximity_features))

            print(x)

            one_hot_vectors = pd.DataFrame(label_binarizer.fit_transform(x["ocean_proximity"]))

            self.preprocessor_params = dict(
                list(zip(ocean_proximity_features, one_hot_vectors.drop_duplicates().values)))
            # print("Dictionary: \n" + str(self.preprocessor_params) + "\n")
        else:
            ocean_proximity_features = list(self.preprocessor_params.keys())
            # print("New features: " + str(ocean_proximity_features))
            one_hot_vectors = pd.DataFrame(
                map(lambda val: self.preprocessor_params[val], x["ocean_proximity"]),
                columns=ocean_proximity_features)
            # print("ELSE CASE ENCODINGS: \n" + str(one_hot_vectors) + "\n")

        encoded_X = pd.concat([x.loc[:, x.columns != "ocean_proximity"], one_hot_vectors], axis=1)
        # print("Encoded Frame: \n" + str(encoded_x) + "\n")

        encoded_X.fillna(0)
        # print("Encoded Frame: \n" + str(encoded_X) + "\n")

        normalised_X = pd.DataFrame(MinMaxScaler().fit_transform(encoded_X),
                                    columns=encoded_X.columns)
        # print("Normalised X: \n" + str(normalised_X) + "\n")

        tensor_X = torch.tensor(normalised_X.values)
        # print("Tensor X: \n" + str(tensor_X) + "\n")

        tensor_Y = torch.tensor(y.values) if isinstance(y, pd.DataFrame) else None

        # print("Tensor Y: " + str(tensor_Y) + "\n")

        return tensor_X, tensor_Y

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

        X, Y = self._preprocessor(x, y=y, training=True)  # Do not forget
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

        X, _ = self._preprocessor(x, training=False)  # Do not forget
        pass

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

        X, Y = self._preprocessor(x, y=y, training=False)  # Do not forget
        return 0  # Replace this code with your own

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
