import numpy as np
from tensorflow import keras

from typing import List, Tuple

def load_mnist() -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Loads the MNIST dataset and returns the training and test sets.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - training_inputs: A 2D NumPy array where each row represents a training input vector.
            - training_labels: A 1D NumPy array where training_labels[i] represents the class label for training_inputs[i].
            - test_inputs: A 2D NumPy array where each row represents a test input vector.
            - test_labels: A 1D NumPy array where test_labels[i] represents the class label for test_inputs[i].
    """
    (training_inputs, training_labels), (test_inputs, test_labels) = keras.datasets.mnist.load_data()
    # Reshape the data to be 2D arrays
    training_inputs = training_inputs.reshape(training_inputs.shape[0], -1)
    test_inputs = test_inputs.reshape(test_inputs.shape[0], -1)

    max_value = np.max(np.abs(training_inputs))
    training_inputs = training_inputs.astype('float32') / max_value
    test_inputs = test_inputs.astype('float32') / max_value

    return training_inputs, training_labels, test_inputs, test_labels



def create_and_train_model(training_inputs: np.ndarray, training_labels: np.ndarray, blocks: int,
                               filter_size: int, filter_number: int, region_size: int,
                               epochs: int, cnn_activation: str) -> keras.models:
    """
    Creates and trains a convolutional neural network model.

    Args:
        training_inputs (np.ndarray): A 2D NumPy array where each row represents a training input vector.
        training_labels (np.ndarray): A 2D NumPy array with a single column where
            training_labels[i, 0] represents the class label for training_inputs[i].
        blocks (int): Specifies how many convolutional layers the model will have.
        filter_size (int): The number of rows of each 2D convolutional filter.
        filter_number (int): The number of 2D convolutional filters in each convolutional layer.
        region_size (int): The size of the region for the max pool layer.
        epochs (int): The number of training epochs.
        hidden_activations (str): A strings specifying the activation function for each hidden layer.
            Possible values are 'tanh', 'sigmoid', and 'relu'.

    Returns:
        keras.model: The trained model.
    """