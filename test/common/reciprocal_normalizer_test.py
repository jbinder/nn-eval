import numpy
import numpy as np
import unittest

from common.reciprocal_normalizer import ReciprocalNormalizer


# noinspection PyMethodMayBeStatic
class ReciprocalNormalizerTest(unittest.TestCase):
    def test_process_valid_data(self):
        normalizer = ReciprocalNormalizer()
        data = np.array([0, 0.5, 0.75, 1, 11])
        expected = np.array([0, 2, 1.3333, 1, 0.0909])
        actual = normalizer.process(data)
        numpy.testing.assert_array_almost_equal(actual, expected, 4)
