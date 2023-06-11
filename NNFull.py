import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, layer_functions=None, learning_rate=0.2):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate

        if not layer_functions:
            layer_functions = ['sigmoid' for _ in layer_sizes]

        self.layers = [Layer(input_size, output_size, function) for output_size, input_size, function in zip(layer_sizes[1:], layer_sizes[:-1], layer_functions)]

    def feed_forward(self, inputs):
        layer_output = inputs

        for layer in self.layers:
            layer_output = layer.feed_through(layer_output)

        return layer_output

    def train(self, inputs, targets):
        layer_outputs = [inputs]

        layer_output = inputs
        for layer in self.layers:
            layer_output = layer.feed_through(layer_output)
            layer_outputs.append(layer_output)

        # Error derivative = target - output
        next_errors = targets - layer_outputs[-1]

        for i in range(1, len(self.layer_sizes)):
            current_values = layer_outputs[-i]
            previous_values = layer_outputs[-(i + 1)]

            # Calculate gradients for current layer
            gradients = self.layers[-i].activation_derivative(current_values) * next_errors

            # Calculate deltas for current layer
            weight_matrix_deltas = np.transpose(np.atleast_2d(gradients)) @ np.atleast_2d(previous_values)

            # Adjust weights by deltas
            self.layers[-i].update_weights(weight_matrix_deltas * self.learning_rate)

            # Adjust biases by deltas - just the hidden gradient
            self.layers[-i].update_bias(gradients * self.learning_rate)

            next_errors = self.layers[-i].back_errors(next_errors)


class Layer:
    def __init__(self, input_size, output_size, function):
        self.weights = np.random.uniform(-1, 1, (output_size, input_size))
        self.bias = np.random.uniform(-1, 1, output_size)

        match function:
            case 'sigmoid':
                self.activation = Layer.sigmoid
                self.activation_derivative = Layer.sigmoid_derivative
            case 'relu':
                self.activation = Layer.relu
                self.activation_derivative = Layer.relu_derivative
            case 'linear':
                self.activation = Layer.linear
                self.activation_derivative = Layer.linear_derivative

    def feed_through(self, input):
        # Calculate weighted sum of inputs + bias
        weighted_sum = self.weights @ input + self.bias

        # Pass output values through activation function
        output = self.activation(weighted_sum)
        return output

    def update_weights(self, delta):
        self.weights += delta

    def update_bias(self, delta):
        self.bias += delta

    def back_errors(self, errors):
        return self.weights.T @ errors

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(y):
        return y * (1 - y)

    @staticmethod
    def relu(x):
        x[x < 0] = 0
        return x

    @staticmethod
    def relu_derivative(y):
        y[y > 0] = 1
        return y

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_derivative(y):
        return 1
