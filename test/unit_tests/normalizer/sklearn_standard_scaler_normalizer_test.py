import numpy

from normalizer.sklearn_standard_normalizer import SklearnStandardNormalizer
# noinspection PyMethodMayBeStatic
from test.unit_tests.normalizer.normalizer_test_base import NormalizerTestBase


class SklearnStandardScalerNormalizerTest(NormalizerTestBase):

    def test_normalize_valid_data(self):
        data = self.source.copy()
        source = self.source.copy()
        normalizer = SklearnStandardNormalizer(data)
        actual = normalizer.normalize(data)
        [self.assertNotAlmostEqual(actual[i], source[i]) for i in range(0, len(source))]
        # assert the input array has not been altered
        numpy.testing.assert_array_almost_equal_nulp(data, source, 4)

    def test_normalize_invalid_data(self):
        data = [0]
        self.assertRaises(Exception, SklearnStandardNormalizer, data)

    def test_denormalize_valid_data(self):
        data = self.source.copy()
        normalizer = SklearnStandardNormalizer(data)
        normalized = normalizer.normalize(data)
        actual = normalizer.denormalize(normalized)
        numpy.testing.assert_array_almost_equal(actual, data, 4)

    def test_denormalize_invalid_data(self):
        data = [0]
        self.assertRaises(Exception, SklearnStandardNormalizer, data)
