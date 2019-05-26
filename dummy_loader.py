import numpy as np


def load_data(max_range, dimension):
    training_data = list()
    test_data = list()

    # input = np.random.randint(max_range, size=dimension)
    input = np.array([
        [1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1],
    ])

    output = np.array([[1]])
    training_data.append(tuple((input, output)))
    test_data.append(tuple((input, output)))

    # input = np.random.randint(max_range, size=dimension)
    input = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    ])
    output = np.array([[1]])
    training_data.append(tuple((input, output)))
    test_data.append(tuple((input, output)))

    return (training_data, test_data)