import numpy
import numpy as np

from normalizer.reciprocal_normalizer import ReciprocalNormalizer
# noinspection PyMethodMayBeStatic
from test.unit_tests.normalizer.normalizer_test_base import NormalizerTestBase


class ReciprocalNormalizerTest(NormalizerTestBase):

    def test_normalize_valid_data(self):
        normalizer = ReciprocalNormalizer()
        source = self.source.copy()
        data = self.source.copy()
        expected = np.array([0, 2, 1.3333, 1, 0, 0.0909])
        actual = normalizer.normalize(data)
        numpy.testing.assert_array_almost_equal(actual, expected, 4)
        # assert the input array has not been altered
        numpy.testing.assert_array_almost_equal_nulp(data, source, 4)

    def test_normalize_invalid_data(self):
        normalizer = ReciprocalNormalizer()
        self.assertRaises(Exception, normalizer.normalize, [0])

    def test_denormalize_valid_data(self):
        normalizer = ReciprocalNormalizer()
        data = self.source.copy()
        normalized = normalizer.normalize(data)
        actual = normalizer.denormalize(normalized)
        numpy.testing.assert_array_almost_equal(actual, data, 4)

    def test_denormalize_invalid_data(self):
        normalizer = ReciprocalNormalizer()
        self.assertRaises(Exception, normalizer.denormalize, [0])
