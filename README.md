# Neural Network Classification with Keras

This repository contains implementations of **Fully Connected (Dense) Neural Networks** and **Convolutional Neural Networks (CNNs)** built with the Keras API in TensorFlow. The project explores fundamental machine learning concepts by applying these models to classic classification problems, including various **UCI datasets** and the **MNIST handwritten digit dataset**.

---

## 🧠 Core Concepts and Theory

### 1. Fully Connected (Dense) Neural Networks
A Dense Network, or Multilayer Perceptron (MLP), is the foundational architecture of neural networks.
* **Structure:** It consists of an input layer, one or more **hidden layers**, and an output layer.
* **Dense Layer:** In a dense layer, every neuron is connected to every neuron in the previous layer. Each connection has an associated **weight**, and each neuron has a **bias**. The output of a neuron is calculated as: `Output` = `Activation(Weighted Sum of Inputs + Bias)`.
* **Application:** These networks are highly effective for **tabular data** (like the UCI datasets) or flattened image data (like the basic MNIST implementation).

### 2. Convolutional Neural Networks (CNNs)
CNNs are specialized neural networks primarily designed for processing data with a known grid-like topology, such as **images**. They exploit the spatial locality of pixels to learn hierarchical feature representations.
* **Convolutional Layer:** This layer uses a small set of trainable weights called a **filter** (or kernel) to scan the input image. It performs a convolution operation, producing a **feature map** that highlights specific features (edges, textures) in the input.
* **Pooling Layer (e.g., Max Pooling):** This layer reduces the dimensionality of the feature map by summarizing regions (e.g., taking the maximum value over a `2x2` region). This makes the model more robust to small variations in the image and reduces computational load.
* **Flatten Layer:** After several Conv/Pool blocks, the 2D or 3D feature maps are "flattened" into a single vector to be passed to a final set of **Dense layers** for classification.

### 3. Activation Functions
Activation functions introduce non-linearity into the network, allowing it to learn complex mappings from inputs to outputs.
* **ReLU (Rectified Linear Unit):** `f(x) = max(0, x)`. The most common choice for hidden layers in deep learning due to its computational efficiency. Used in the CNN implementation.
* **Tanh (Hyperbolic Tangent):** `f(x) = (e^x - e^{-x}) / (e^x + e^{-x})`. Outputs values between -1 and 1.
* **Sigmoid:** `f(x) = 1 / (1 + e^{-x})`. Outputs values between 0 and 1. Historically used, but often leads to the vanishing gradient problem in deep layers.
* **Softmax (Output Layer):** Used in the final layer for **multi-class classification**. It converts the raw output scores (logits) into a probability distribution, where all probabilities sum up to 1.

### 4. Loss Function and Optimization
* **Loss Function (SparseCategoricalCrossentropy):** Measures the difference between the model's predicted probabilities and the true target label. **Sparse** is used here because the true labels are provided as integer indices (e.g., 0, 1, 2) rather than one-hot encoded vectors. The goal of training is to minimize this loss.
* **Optimizer (Adam):** An algorithm used to adjust the network's weights during training to minimize the loss function. Adam is an efficient and popular optimization algorithm.
* **Epochs:** One complete pass of the entire training dataset through the neural network.

---

## 📂 Project Structure and File Descriptions

| File Name | Description | Key Components |
| :--- | :--- | :--- |
| **`uci_load.py`** | Helper utility to read and preprocess generic classification datasets from the UCI Machine Learning Repository format. It handles label-to-integer mapping. | `read_uci_file()`, `read_uci_dataset()` |
| **`nn_keras_solution.py`** | Contains the functions to build and train a **Dense Network** and test its performance on generic UCI datasets. | `create_and_train_model()`, `test_model()` |
| **`nn_keras_base.py`** | The main execution script for the **UCI dataset** experiments. It loads the data using `uci_load.py`, normalizes the inputs, and runs the training process multiple times to calculate mean/std accuracy. | Uses `yeast_string` dataset by default. |
| **`dense_mnist_solution.py`** | Contains functions for loading the **MNIST** dataset and implementing a **Dense Network** on the flattened image data. | `load_mnist()`, `create_and_train_model()` (Dense layers only) |
| **`dense_mnist_base.py`** | The main execution script for the **Dense MNIST** experiment. Loads the data, sets model parameters (layers, units, epochs, activations), and runs training iterations. | Implements `keras.layers.Flatten` to prepare the image for the Dense network. |
| **`cnn_mnist_solution.py`** | Contains functions for loading the **MNIST** dataset and implementing a **Convolutional Neural Network (CNN)**. | Includes `keras.layers.Conv2D`, `keras.layers.MaxPooling2D`, and appropriate reshaping for channel dimension. |
| **`cnn_mnist_base.py`** | The main execution script for the **CNN MNIST** experiment. Configures the CNN architecture (blocks, filter size, number) and runs training iterations. | Demonstrates the superior performance of CNNs for image tasks. |

---

## 🛠 Installation and Requirements

To run the experiments, you need a standard Python environment with NumPy and TensorFlow/Keras installed.

1.  **Clone the repository:**
    `git clone https://github.com/Amxgh/ConvolutionalNeuralNetworks`
2.  **Install dependencies:**
    `pip install tensorflow numpy`

*(Note: TensorFlow will automatically install Keras.)*

## 🚀 Running the Experiments

The project is structured so that you can run the *base* files to execute the corresponding experiments.

### 1. Generic Classification (UCI Datasets)

This task uses Dense Networks for generic classification.

* **Setup:** Ensure you have the corresponding UCI dataset files (e.g., `yeast_string_training.txt`, `yeast_string_test.txt`) in a subdirectory named `uci_datasets`.
* **Execution:**
    `python nn_keras_base.py`
* **Configuration:** You can modify the following parameters in `nn_keras_base.py`:
    * `dataset`: Change the UCI dataset name.
    * `layers`, `units_per_layer`, `hidden_activations`: Adjust the network's size and activation functions.
    * `epochs`: Change the training duration.

### 2. MNIST Classification (Dense Network)

This is a baseline comparison, flattening the `28x28` image into a `784`-feature vector for a Dense Network.

* **Execution:**
    `python dense_mnist_base.py`
* **Key Feature:** The `dense_mnist_solution.py` file handles the MNIST data loading and uses `keras.layers.Flatten` before adding Dense layers.

### 3. MNIST Classification (Convolutional Network)

This is the standard and most performant approach for image classification on MNIST.

* **Execution:**
    `python cnn_mnist_base.py`
* **Configuration:** You can adjust the CNN architecture in `cnn_mnist_base.py`:
    * `blocks`: Number of (Conv -> MaxPool) blocks.
    * `filter_size`, `filter_number`, `region_size`: Define the kernel, depth, and pooling dimensions.

---
