import numpy as np


def nonlin(x, deriv=False):
    if (deriv == True):
        return nonlin(x) * (1 - nonlin(x))
    return 1 / (1 + np.exp(-x))

def get_weights(cycles_count, train_data, train_output):
    np.random.seed(1)
    weights = 2 * np.random.random((3, 1)) - 1

    for iter in range(cycles_count):
        l1 = nonlin(np.dot(train_data, weights))
        l1_error = (train_output - l1)**2
        l1_delta = l1_error * nonlin(l1, True)
        weights += np.dot(train_data.T, l1_delta)

    return weights

def get_neural_output(data, weights):
    return nonlin(np.dot(data, weights))


