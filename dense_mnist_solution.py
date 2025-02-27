from typing import List, Tuple

import numpy as np
from tensorflow import keras


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


