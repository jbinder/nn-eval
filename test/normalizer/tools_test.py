import unittest

import numpy as np

from normalizer.tools import get_normalizer


# noinspection PyMethodMayBeStatic
class ToolsTest(unittest.TestCase):
    def test_normalize_valid_data(self):
        normalizer_names = ['Identity', 'Reciprocal', 'SklearnStandard']
        for normalizer_name in normalizer_names:
            normalizer = get_normalizer(normalizer_name, np.array([1, 2, 3]))
            self.assertIsNotNone(normalizer)
