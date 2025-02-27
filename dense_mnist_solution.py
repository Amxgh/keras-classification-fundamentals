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



def create_and_train_model(training_inputs: np.ndarray, training_labels: np.ndarray, layers: int,
                           units_per_layer: List[int], epochs: int, hidden_activations: List[str]) -> keras.models:
    """
    Creates and trains a neural network model.

    Args:
        training_inputs (np.ndarray): A 2D NumPy array where each row represents a training input vector.
        training_labels (np.ndarray): A 2D NumPy array with a single column where
            training_labels[i, 0] represents the class label for training_inputs[i].
        layers (int): The total number of layers in the neural network.
        units_per_layer (List[int]): A list specifying the number of units in each hidden layer.
            The length of this list should be layers - 2.
            For example, units_per_layer[0] corresponds to the number of units in the first hidden layer.
        epochs (int): The number of training epochs.
        hidden_activations (List[str]): A list of strings specifying the activation function for each hidden layer.
            Possible values are 'tanh', 'sigmoid', and 'relu'.

    Returns:
        object: The trained model.
    """

    # Let Keras use its default method for initialization of all weights (i.e., your code should not address this issue at all).

    input_shape = training_inputs[0].shape
    number_of_classes = np.max([np.max(training_labels)]) + 1

    # model = keras.Sequential(keras.Input(shape=input_shape), keras.layers.Dense(number_of_classes, activation='sigmoid'))

    # model.add(keras.layers.Dense(units_per_layer[0], input_dim=training_inputs.shape[1], activation=hidden_activations[0]))

    model_input = [keras.Input(shape=input_shape)]

    for units, activation in zip(units_per_layer, hidden_activations):
        model_input.append(keras.layers.Dense(units=units, activation=activation))

    model_input.append(keras.layers.Dense(number_of_classes, activation='sigmoid'))

    model = keras.Sequential(model_input)
    #
    # for units, activation in zip(units_per_layer, hidden_activations):
    #     model.add(keras.layers.Dense(units=units, activation=activation))
    #
    # model.add(keras.layers.Dense(number_of_classes, activation='softmax'))
    #
    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    model.fit(training_inputs, training_labels, epochs=epochs)

    return model