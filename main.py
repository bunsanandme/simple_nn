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
        l1_error = train_output - l1
        l1_delta = l1_error * nonlin(l1, True)
        weights += np.dot(train_data.T, l1_delta)

    return weights

def get_neural_output(data, weights):
    return nonlin(np.dot(data, weights))

# (A or C) or not B
train_input = np.array([[1, 0, 1],
              [0, 0, 1],
              [0, 1, 0],
              [1, 0, 0]])

train_output = np.array([[1, 1, 0, 1]]).T

weights = get_weights(100000, train_input, train_output)

test = np.array([0,0,0])
print("Test input: ", test)

test_output = get_neural_output(test, weights)

print("Test output: ", np.round(test_output,0))

