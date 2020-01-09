import numpy as np

from normalizer.normalizer_base import NormalizerBase


class ReciprocalNormalizer(NormalizerBase):
    """ Normalizes values in a numpy array using the reciprocal value. """

    def normalize(self, data):
        data = NormalizerBase.normalize(self, data)
        return ReciprocalNormalizer._calculate_reciprocal(data)

    def denormalize(self, data):
        data = NormalizerBase.denormalize(self, data)
        return ReciprocalNormalizer._calculate_reciprocal(data)

    @staticmethod
    def _calculate_reciprocal(data):
        # TODO: ensure the output lies between 0 and 1
        # while np.any((data != 0) & (data < 1)):
        #     data = np.where(data >= 1, data, data * 10)
        result = np.copy(data)
        result[data != 0] = 1 / data[data != 0]
        return result
