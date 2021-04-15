import numpy as np

from layered.network import Network, Layer
from layered.activation import Identity, Relu, Softmax

num_inputs = 784
num_outputs = 10

network = Network([
    Layer(num_inputs, Identity),
    Layer(700, Relu),
    Layer(500, Relu),
    Layer(300, Relu),
    Layer(num_outputs, Softmax),
])

from layered.network import Matrices

weight_scale = 0.01

weights = Matrices(network.shapes)
weights.flat = np.random.normal(0, weight_scale, len(weights.flat))

from layered.cost import SquaredError
from layered.gradient import Backprop
from layered.optimization import GradientDecent

backprop = Backprop(network, cost=SquaredError())
descent = GradientDecent()

from layered.dataset import Mnist

dataset = Mnist()
for example in dataset.training:
    gradient = backprop(weights, example)
    weights = descent(weights, gradient, learning_rate=0.1)

error = 0
for example in dataset.testing:
    prediction = network.feed(weights, example.data)
    if np.argmax(prediction) != np.argmax(example.target):
        error += 1 / len(dataset.testing)
print('Testing error', round(100 * error, 2), '%')