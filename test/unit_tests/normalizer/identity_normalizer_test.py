import numpy

from normalizer.identity_normalizer import IdentityNormalizer
# noinspection PyMethodMayBeStatic
from test.unit_tests.normalizer.normalizer_test_base import NormalizerTestBase


class ReciprocalNormalizerTest(NormalizerTestBase):

    def test_normalize_valid_data(self):
        normalizer = IdentityNormalizer()
        source = self.source.copy()
        data = self.source.copy()
        actual = normalizer.normalize(data)
        numpy.testing.assert_array_almost_equal(actual, source, 4)
        # assert the input array has not been altered
        numpy.testing.assert_array_almost_equal_nulp(data, source, 4)

    def test_normalize_invalid_data(self):
        normalizer = IdentityNormalizer()
        self.assertRaises(Exception, normalizer.normalize, [0])

    def test_denormalize_valid_data(self):
        normalizer = IdentityNormalizer()
        data = self.source.copy()
        normalized = normalizer.normalize(data)
        actual = normalizer.denormalize(normalized)
        numpy.testing.assert_array_almost_equal(actual, data, 4)

    def test_denormalize_invalid_data(self):
        normalizer = IdentityNormalizer()
        self.assertRaises(Exception, normalizer.denormalize, [0])
