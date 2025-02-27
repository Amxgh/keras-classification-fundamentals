from typing import List

import numpy as np
from tensorflow import keras


def create_and_train_model(training_inputs: np.ndarray, training_labels: np.ndarray, layers: int,
                           units_per_layer: List[int], epochs: int,
                           hidden_activations: List[str]) -> keras.models:
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
        keras.models: The trained model.
    """

    input_shape = training_inputs[0].shape
    number_of_classes = np.max([np.max(training_labels)]) + 1

    model_input = [keras.Input(shape=input_shape)]

    for units, activation in zip(units_per_layer, hidden_activations):
        model_input.append(keras.layers.Dense(units=units, activation=activation))

    model_input.append(keras.layers.Dense(number_of_classes, activation='sigmoid'))

    model = keras.Sequential(model_input)

    model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    model.fit(training_inputs, training_labels, epochs=epochs)

    return model


def test_model(model: keras.models, test_inputs: np.ndarray, test_labels: np.ndarray,
               ints_to_labels: dict) -> float:
    """
    Evaluates the trained model on the test set.

    Args:
        model (keras.models): The trained model.
        test_inputs (np.ndarray): 2D numpy array where each row is a test input vector.
        test_labels (np.ndarray): numpy column vector. 2D numpy array with a single column  where
            test_labels[i, 0] represents the class label for test_inputs[i].
        ints_to_labels (dict): Maps int labels to the original class labels

    """
    accuracy = []

    for i in range(len(test_inputs)):
        pred = model.predict(test_inputs[i].reshape(1, -1), verbose=0)

        true_class = ints_to_labels[test_labels[i, 0]]

        has_duplicate = np.count_nonzero(pred == np.argmax(pred))

        pred_class = ints_to_labels[np.argmax(pred)]
        if has_duplicate > 1:
            for x in np.where(pred == np.argmax(pred))[0]:
                if true_class == ints_to_labels[x]:
                    pred_class = ints_to_labels[x]
                    break

            accuracy.append(int(true_class == pred_class) / len(np.where(pred == np.argmax(pred))[0]))
        else:
            accuracy.append(int(true_class == pred_class))

        pred_class = ints_to_labels[np.argmax(pred)]
        accuracy.append(pred_class == true_class)

        print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f' %
              (i, pred_class, true_class, accuracy[i]))

    test_loss, test_accuracy = model.evaluate(test_inputs, test_labels, verbose=0)

    return test_accuracy
