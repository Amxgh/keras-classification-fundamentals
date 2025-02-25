from typing import List
from tensorflow import keras


import numpy as np


def create_and_train_model(training_inputs: np.ndarray, training_labels: np.ndarray, layers: int,
                           units_per_layer: List[int], epochs: int, hidden_activations: List[str]) -> object:
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
    model = keras.Sequential()
    # model.add(keras.layers.Dense(units_per_layer[0], input_dim=training_inputs.shape[1], activation=hidden_activations[0]))

    for units, activation in zip(units_per_layer, hidden_activations):
        model.add(keras.layers.Dense(units=units, activation=activation))

    number_of_classes = len(np.unique(training_labels))
    model.add(keras.layers.Dense(number_of_classes, activation='softmax'))

    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    model.fit(training_inputs, training_labels, epochs=epochs, verbose=1)

    return model


def test_model(model: object, test_inputs: np.ndarray, test_labels: np.ndarray, ints_to_labels:dict) -> float:
    """
    Evaluates the trained model on the test set.

    Args:
        model (object): The trained model.
        test_inputs (np.ndarray): 2D numpy array where each row is a test input vector.
        test_labels (np.ndarray): numpy column vector. 2D numpy array with a single column  where
            test_labels[i, 0] represents the class label for test_inputs[i].
        ints_to_labels (dict): Maps int labels to the original class labels

    """
    pass
