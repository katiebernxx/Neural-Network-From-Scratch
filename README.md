# MultiLayer Perceptron Implementation Without Deep Learning Libraries

This project demonstrates a deep understanding of the architecture and functioning of a multilayer perceptron
by building it from scratch without the use of deep learning libraries. Heavy comments throughout explain 
how the multilayer perceptron is actually working and how the math behind the scenes works to run and train the
model. Assertions of the matrix sizes show an understanding of the matrix math.

## Table of Contents

1. [Project Structure](#Project-Structure)
2. [Usage](#Usage)
3. [Key Concepts](#key-concepts)

## Project Structure
### Initialization of Parameters:
* Function: init_params(layer_sizes)
* Purpose: Initializes weights and biases for each layer using He initialization to prevent vanishing or exploding gradients.

### Activation Functions
* Functions: sigmoid(x, derivative=False), softmax(x, derivative=False, epsilon=1e-12), ReLU(x, derivative=False)
* Sigmoid Function: Used in output layers for binary classification.
* Softmax Function: Used in output layers for multi-class classification.
* ReLU Function: Commonly used in hidden layers to address the vanishing gradient problem.

### Creating Batches
* Function: create_batches(x_train, y_train, batch_size)
* Purpose: Splits the training data into mini-batches for mini-batch gradient descent.

### Forward Pass
* Function: forward_pass(x_batch, params, layer_sizes)
* Purpose: Propagates input data through the network to generate output.
* Comments: Uses ReLU for hidden layers and Softmax for the output layer.

### Loss Calculations
* Functions: categorical_cross_entropy_loss_gradient(y_true, y_pred, epsilon=1e-12), compute_loss(y_true, y_pred, epsilon=1e-12)
* Purpose: Computes the loss using categorical cross-entropy for classification tasks. Additionally to show loss in each epoch during training.

### Backward Pass
* Function: backward_pass(y_batch, params, layer_sizes)
* Purpose: Computes gradients of the loss with respect to each parameter using the chain rule to propagate the error backwards through the network.

### Updating Parameters
* Function: update_parameters(params, grads, layer_sizes, learning_rate=0.01)
* Purpose: Updates the weights and biases based on the gradients computed during backpropagation.

### Training The Network
* Function: train(x_train, y_train, params, layer_sizes, epochs, batch_size, learning_rate)
* Purpose: Trains the neural network using mini-batch gradient descent by looping through forward and backward passes.

### Making Predictions
* Function: predict(x, params, layer_sizes)
* Purpose: Uses the trained model to make predictions on new data.

## Usage
### Initialization
* Define the architecture of the network by specifying the number of neurons in each layer.
* Initialize parameters using init_params(layer_sizes).

### Training
* Train the network by calling train(x_train, y_train, params, layer_sizes, epochs, batch_size, learning_rate).

### Prediction
* Make predictions on new data using predict(x, params, layer_sizes).

## Key Concepts
* He Initialization: Used to initialize weights to help with gradient issues in deep networks.
* Activation Functions: Introduce non-linearity, essential for learning complex patterns.
* Mini-Batch Gradient Descent: Combines the benefits of both batch and stochastic gradient descent for efficient and stable training.
* Forward and Backward Passes: Core components of neural network training, involving propagating inputs forward and errors backward through the network.
* Categorical Cross-Entropy Loss: Measures the performance of a classification model whose output is a probability value between 0 and 1.
