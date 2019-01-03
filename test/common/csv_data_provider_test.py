import unittest

from common.csv_data_provider import CsvDataProvider


class OptimizerTest(unittest.TestCase):
    def test_get_data_from_file_valid_input(self):
        provider = CsvDataProvider()
        data = provider.get_data_from_file("../../data/001_linear/x.csv", "../../data/001_linear/y.csv")
        self.assertGreater(len(data['train'][0]), 0)
        self.assertGreater(len(data['train'][1]), 0)
        self.assertGreater(len(data['valid'][0]), 0)
        self.assertGreater(len(data['valid'][1]), 0)
        self.assertGreaterEqual(len(data['train'][0]), len(data['valid'][0]))
        self.assertGreaterEqual(len(data['train'][1]), len(data['valid'][1]))
