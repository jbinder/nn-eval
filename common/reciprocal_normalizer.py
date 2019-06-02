import numpy as np


class ReciprocalNormalizer:
    """ Normalizes values in a numpy array using the reciprocal value. """

    @staticmethod
    def normalize(data):
        if type(data) is not np.ndarray:
            raise ValueError("data needs to be a numpy array.")
        return ReciprocalNormalizer._calculate_reciprocal(data)

    @staticmethod
    def denormalize(data):
        return ReciprocalNormalizer.normalize(data)

    @staticmethod
    def _calculate_reciprocal(data):
        # TODO: ensure the output lies between 0 and 1
        # while np.any((data != 0) & (data < 1)):
        #     data = np.where(data >= 1, data, data * 10)
        result = np.copy(data)
        result[data != 0] = 1 / data[data != 0]
        return result
