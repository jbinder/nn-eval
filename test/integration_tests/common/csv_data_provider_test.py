import os
import unittest

from common import tools
from common.csv_data_provider import CsvDataProvider


class OptimizerTest(unittest.TestCase):
    def test_get_data_from_file_valid_input(self):
        provider = CsvDataProvider()
        x = os.path.join(tools.get_root_dir(), "data", "001_linear", "x.csv")
        y = os.path.join(tools.get_root_dir(), "data", "001_linear", "y.csv")
        data = provider.get_data_from_file(x, y)
        self.assertGreater(len(data['train'][0]), 0)
        self.assertGreater(len(data['train'][1]), 0)
        self.assertGreater(len(data['valid'][0]), 0)
        self.assertGreater(len(data['valid'][1]), 0)
        self.assertGreaterEqual(len(data['train'][0]), len(data['valid'][0]))
        self.assertGreaterEqual(len(data['train'][1]), len(data['valid'][1]))
