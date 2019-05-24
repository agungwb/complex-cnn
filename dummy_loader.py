import numpy as np


def load_data(max_range, dimension):
    training_data = list()
    test_data = list()

    input = np.random.randint(max_range, size=dimension)
    output = np.array([[1]])
    training_data.append(tuple((input, output)))
    test_data.append(tuple((input, output)))

    input = np.random.randint(max_range, size=dimension)
    output = np.array([[1]])
    training_data.append(tuple((input, output)))
    test_data.append(tuple((input, output)))

    return (training_data, test_data)
