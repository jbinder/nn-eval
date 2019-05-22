import numpy as np


class ReciprocalNormalizer:
    """ Normalizes values in a numpy array using the reciprocal value. """

    @staticmethod
    def process(data):
        if type(data) is not np.ndarray:
            raise ValueError("data needs to be a numpy array.")
        # TODO: ensure the output lies between 0 and 1
        # while np.any((data != 0) & (data < 1)):
        #     data = np.where(data >= 1, data, data * 10)
        data = 1 / data
        data[data == np.inf] = 0
        return data
