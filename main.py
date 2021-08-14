from NeuralNetwork import *
import matplotlib.pyplot as mpl



if __name__ == "__main__":
    # (A or C) or not B
    train_input = np.array([[1, 0, 1],
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]])

    train_output = np.array([[1, 1, 0, 1]]).T

    network = NeuralNetwork(1)
    weights = network.train(train_input, train_output)
    mpl.plot(network.get_errors())
    mpl.show()
    # print("Test output: ", np.round(train_output, 0).T)