import numpy
import numpy as np
import unittest

from normalizer.identity_normalizer import IdentityNormalizer
from normalizer.reciprocal_normalizer import ReciprocalNormalizer


# noinspection PyMethodMayBeStatic
class ReciprocalNormalizerTest(unittest.TestCase):
    def test_normalize_valid_data(self):
        normalizer = IdentityNormalizer()
        source = np.array([0, 0.5, 0.75, 1, 0, 11])
        data = np.array([0, 0.5, 0.75, 1, 0, 11])
        actual = normalizer.normalize(data)
        numpy.testing.assert_array_almost_equal(actual, source, 4)
        # assert the input array has not been altered
        numpy.testing.assert_array_almost_equal_nulp(data, source, 4)

    def test_normalize_invalid_data(self):
        normalizer = IdentityNormalizer()
        self.assertRaises(Exception, normalizer.normalize, [0])

    def test_denormalize_valid_data(self):
        normalizer = IdentityNormalizer()
        data = np.array([0, 0.5, 0.75, 1, 11])
        normalized = normalizer.normalize(data)
        actual = normalizer.denormalize(normalized)
        numpy.testing.assert_array_almost_equal(actual, data, 4)

    def test_denormalize_invalid_data(self):
        normalizer = IdentityNormalizer()
        self.assertRaises(Exception, normalizer.denormalize, [0])

