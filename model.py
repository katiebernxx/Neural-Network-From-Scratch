""" Building a MultiLayer Perceptron Without The Use of Deep Learning Libraries
    This code will show a deep understanding of the architecture and functionings of a neural network.
    Katie Bernard """

import numpy as np
import matplotlib.pyplot as plt

""" Initializing the parameters (weight and biases) """


def init_params(layer_sizes):
    """
    Initializes the parameters by creating a dictionary with an entry for each layer's weights and biases.

    Args:
    layer_sizes -- list of integers, specifying the number of neurons in each layer

    Returns:
    params -- dictionary containing the initialized weights and biases for each layer
    """
    params = {}
    L = len(layer_sizes)
    for l in range(1, L):
        params["W" + str(l)] = np.random.randn(
            layer_sizes[l], layer_sizes[l - 1]
        ) * np.sqrt(2.0 / layer_sizes[l - 1])
        # We initialize the weights with He Initialization to prevent vanishing or exploding gradients
        # This is similar to Xavier Initialization but suitable for ReLU activations
        params["b" + str(l)] = np.zeros((layer_sizes[l], 1))
        # We initialize the biases to 0
        assert params["W" + str(l)].shape == (
            layer_sizes[l],
            layer_sizes[l - 1],
        ), f"Weight shape mismatch at layer {l}"
        assert params["b" + str(l)].shape == (
            layer_sizes[l],
            1,
        ), f"Bias shape mismatch at layer {l}"

    return params


""" Building the activation functions
    Activation functions introduce nonlinearity to the model.
    They must be injective, differentiable, and continuous for smooth backpropagation """


# The Sigmoid function squashes numbers to the range of [0,1] here.
# It is used in the output layer for binary or multi-class classification.
# It is not often used in hidden layers for risk of vanishing gradients
def sigmoid(x, derivative=False):
    """
    Defines a sigmoid activation function and its derivative.

    Args:
    x -- input numpy array
    derivative -- boolean, if True returns the derivative of the sigmoid function

    Returns:
    result -- numpy array after applying the sigmoid function or its derivative
    """
    if derivative:
        return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
    return 1 / (1 + np.exp(-x))


# The Softmax function normalizes a vector of numbers over a probability distribution.
# It is used in the output layer for multi-class classification problems
def softmax(x, derivative=False, epsilon=1e-12):
    """
    Defines a softmax activation function and its derivative.

    Args:
    x -- input numpy array
    derivative -- boolean, if True returns the derivative of the softmax function

    Returns:
    softmax_vals -- numpy array after applying the softmax function or its derivative
    """
    x = np.clip(x, -1e12, 1e12)  # Clip values to prevent overflow
    shift_x = x - np.max(x, axis=0, keepdims=True)
    exps = np.exp(shift_x)
    softmax_vals = exps / (np.sum(exps, axis=0, keepdims=True) + epsilon)
    if derivative:
        return softmax_vals * (1 - softmax_vals)
    return softmax_vals


# The ReLU function is linear in the positive direction and 0 in the negative direction.
# This helps solve the vanishing gradient problem and allows models to learn faster and perform better
# It is often used in the hidden layers
def ReLU(x, derivative=False):
    """
    Defines a ReLU activation function and its derivative.

    Args:
    x -- input numpy array
    derivative -- boolean, if True returns the derivative of the ReLU function

    Returns:
    result -- numpy array after applying the ReLU function or its derivative
    """
    if derivative:
        return np.where(x > 0, 1, 0)
    else:
        return np.maximum(0, x)


""" There are many more activation functions including Leaky ReLU, Tanh, ELU """


""" Creating batches 
    Mini-batch gradient decsent updates parameters during training after each mini-batch rather than after each
    induvidual sample (SGD) or after the entirty of the training data (regular gradient descent). This 
    increases convergence time, helps the model escape from local minima, and is more computationally effective 
    than regular gradient descent while being more stable and smooth than stochastic gradient descent """


def create_batches(x_train, y_train, batch_size):
    """
    Splits the training data into mini-batches.

    Args:
    x_train -- training data features, numpy array of shape (number of examples, number of features)
    y_train -- training data labels, numpy array of shape (number of examples, number of classes)
    batch_size -- size of each mini-batch

    Returns:
    x_batches -- list of numpy arrays, each containing a mini-batch of features
    y_batches -- list of numpy arrays, each containing a mini-batch of labels
    """
    num_batches = x_train.shape[0] // batch_size
    x_batches = np.array_split(x_train, num_batches)
    y_batches = np.array_split(y_train, num_batches)
    for xb, yb in zip(x_batches, y_batches):
        assert xb.shape[1] == x_train.shape[1], "Feature mismatch in batches"
        assert yb.shape[1] == y_train.shape[1], "Label mismatch in batches"
    return x_batches, y_batches


""" Propagating Forwards 
    The forward pass takes input to the model and performs the calculations forward through the
    neurons of the network to generate an output. 
    
    To calculate the z values of the neurons in a given layer, you dot product the layer weights
    by the previous layer's activations and add the matrix of biases. You get the activations of
    those neurons by applying the activation function to the z values. """


def forward_pass(x_batch, params, layer_sizes):
    """
    Performs the forward pass through the network.

    Args:
    x_batch -- input data batch, numpy array of shape (number of examples in batch, number of features)
    params -- dictionary containing the weights and biases of the network
    layer_sizes -- list of integers, specifying the number of neurons in each layer

    Returns:
    output -- activations of the output layer, numpy array of shape (number of classes, number of examples in
    batch)
    """
    # Ensure input is in the correct shape
    if x_batch.shape[0] != layer_sizes[0]:
        x_batch = x_batch.T
    assert (
        x_batch.shape[0] == layer_sizes[0]
    ), "Input features do not match the network's input size"

    # Activations of the first layer are just the inputs
    params["A0"] = x_batch

    L = len(layer_sizes) - 1  # Number of layers

    # Updating Z values and Activations for each hidden layer (using ReLU)
    for l in range(1, L):
        params["Z" + str(l)] = (
            np.dot(params["W" + str(l)], params["A" + str(l - 1)])
            + params["b" + str(l)]
        )
        params["A" + str(l)] = ReLU(params["Z" + str(l)])
        assert params["Z" + str(l)].shape == (
            layer_sizes[l],
            x_batch.shape[1],
        ), f"Shape mismatch at Z{l}"

    # Output layer (using softmax for multi-class classification)
    params["Z" + str(L)] = (
        np.dot(params["W" + str(L)], params["A" + str(L - 1)]) + params["b" + str(L)]
    )
    params["A" + str(L)] = softmax(params["Z" + str(L)])
    assert params["A" + str(L)].shape[0] == layer_sizes[L], "Output shape mismatch"

    # Return the output of the final layer
    return params["A" + str(L)]


""" Loss Calculation
    Once we have completed a forward pass, we need to calculate the loss achieved from the
    pass with those weights and biases. We will build a categorical cross-entropy loss function
    because it will measure the performance of a classification model that outputs probability values
    from 0 - 1. In our model, we use the softmax activation function in the output layer to do that.
    """


def categorical_cross_entropy_loss_gradient(y_true, y_pred, epsilon=1e-12):
    """
    Computes the gradient of the categorical cross-entropy loss function.

    Args:
    y_true -- true labels, one-hot encoded numpy array of shape (number of classes, number of examples)
    y_pred -- predicted probabilities, numpy array of shape (number of classes, number of examples)

    Returns:
    dA -- gradient of the loss with respect to the activation output, same shape as y_pred
    """
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    dA = -(y_true / y_pred)
    return dA


def compute_loss(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]
    return loss


""" Backwards Pass
    The backwards pass computes the gradients of the high dimensional loss function with respect to the current 
    parameters. We apply the chain rule to propogate the error backwards through the network, calculating the
    gradient of loss with respect to each weight.
    
    For each neuron in the output layer, we calciulate the error gradient with respect to the output. For a 
    hidden layer neuron, the gradient is calculated as the sum of the gradients in a subsequent layer, 
    multiplied by the derivative of the activation function
    """


def backward_pass(y_batch, params, layer_sizes):
    """
    Performs the backward pass through the network to compute gradients.

    Args:
    y_batch -- true labels for the batch, numpy array of shape (number of examples in batch, number of classes)
    params -- dictionary containing the weights and biases of the network
    layer_sizes -- list of integers, specifying the number of neurons in each layer

    Returns:
    grads -- dictionary containing the gradients of the weights and biases
    """
    grads = {}  # Dictionary to store the gradients
    L = len(layer_sizes) - 1  # Number of layers (excluding the input layer)

    m = y_batch.shape[0]  # Number of examples in the batch

    # Calculate dA for the final layer using categorical cross-entropy loss gradient
    params["dA" + str(L)] = categorical_cross_entropy_loss_gradient(
        y_batch.T, params["A" + str(L)]
    )
    assert (
        params["dA" + str(L)].shape == params["A" + str(L)].shape
    ), "Gradient shape mismatch at output layer"

    # Iterate backward from the last layer to the first hidden layer
    for l in reversed(range(1, L + 1)):
        # Compute dZ for layer l: derivative of the cost with respect to Z^l (Z^l is the pre-activation value)
        params["dZ" + str(l)] = params["dA" + str(l)] * ReLU(
            params["Z" + str(l)], derivative=True
        )
        # Compute dW for layer l: gradient of the cost with respect to W^l
        grads["dW" + str(l)] = (1 / m) * np.dot(
            params["dZ" + str(l)], params["A" + str(l - 1)].T
        )
        # Compute db for layer l: gradient of the cost with respect to b^l
        grads["db" + str(l)] = (1 / m) * np.sum(
            params["dZ" + str(l)], axis=1, keepdims=True
        )
        # Compute dA for the previous layer l-1 if we are not at the first layer
        if l > 1:
            params["dA" + str(l - 1)] = np.dot(
                params["W" + str(l)].T, params["dZ" + str(l)]
            )
            assert (
                params["dA" + str(L)].shape == params["A" + str(L)].shape
            ), "Gradient shape mismatch at output layer"

    return grads


""" Updating parameters
    Update the parameters based on gradients computed during the backwards pass in an effort to optimize the 
    high dimensional cost function. We take steps of size and direction (gradient of C * learning rate) and 
    progress towards a minimum of the cost function  """


def update_parameters(params, grads, layer_sizes, learning_rate=0.01):
    """
    Updates the weights and biases of the network using mini batch gradient descent.

    Args:
        params -- Dictionary containing the current weights and biases of the network.
        grads -- Dictionary containing the gradients of the weights and biases.
        layer_sizes -- List of integers specifying the number of neurons in each layer.
        learning_rate -- The learning rate for gradient descent.

    Returns:
        params -- Dictionary containing the updated weights and biases.
    """
    L = len(layer_sizes) - 1  # Number of layers
    for l in range(1, L + 1):
        # Update the weights for layer l
        params["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        # Update the biases for layer l
        params["b" + str(l)] -= learning_rate * grads["db" + str(l)]
        assert params["W" + str(l)].shape == (
            layer_sizes[l],
            layer_sizes[l - 1],
        ), f"Weight update shape mismatch at layer {l}"
        assert params["b" + str(l)].shape == (
            layer_sizes[l],
            1,
        ), f"Bias update shape mismatch at layer {l}"

    return params  # Return the updated parameters


""" Training the network. 
    Loop through the forward pass, backward pass, and gradient descent to update the parameters with mini-
    batches."""


def train(x_train, y_train, params, layer_sizes, epochs, batch_size, learning_rate):
    """
    Trains the neural network using mini-batch gradient descent.

    Args:
        x_train -- Array of training data features.
        y_train -- Array of training data labels.
        params --  Dictionary containing the initial weights and biases of the network.
        layer_sizes -- List of integers specifying the number of neurons in each layer.
        epochs -- Number of training epochs.
        batch_size -- Size of each mini-batch.
        learning_rate -- The learning rate for gradient descent.

    Returns:
        params -- Dictionary containing the updated weights and biases after training.
    """
    for epoch in range(epochs):
        total_loss = 0

        x_batches, y_batches = create_batches(x_train, y_train, batch_size)

        for x_batch, y_batch in zip(x_batches, y_batches):
            # Forward pass
            output = forward_pass(x_batch, params, layer_sizes)

            # Compute loss
            loss = compute_loss(y_batch.T, output)
            total_loss += loss

            # Backward pass
            grads = backward_pass(y_batch, params, layer_sizes)

            # Update parameters
            params = update_parameters(params, grads, layer_sizes, learning_rate)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(x_batches)}")

    return params


def predict(x, params, layer_sizes):
    """
    Makes predictions using the trained neural network.

    Args:
        x -- Array of input data features.
        params -- Dictionary containing the trained weights and biases of the network.
        layer_sizes -- List of integers specifying the number of neurons in each layer.

    Returns:
        predictions -- Array of predicted class labels for the input data.
    """
    # Ensure input data is correctly shaped
    if x.shape[0] != layer_sizes[0]:
        x = x.T
    output = forward_pass(x, params, layer_sizes)
    predictions = np.argmax(output, axis=0)
    return predictions