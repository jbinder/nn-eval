import unittest

import torch
import torch.utils.data as utils_data

import main


class PtbTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_run_linear(self):
        data_loader = {}
        dataset = utils_data.TensorDataset(
            torch.FloatTensor([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            torch.FloatTensor([3, 6, 9, 12, 15, 18, 21, 24, 27]))
        data_loader['train'] = utils_data.DataLoader(dataset)
        dataset = utils_data.TensorDataset(
            torch.FloatTensor([10, 11, 12]),
            torch.FloatTensor([30, 33, 36]))
        data_loader['valid'] = utils_data.DataLoader(dataset)
        train_options = main.TrainOptions(50, 100, True)
        network_options = main.NetworkOptions(1, 1, [4, 8])
        loss = main.run(data_loader, network_options, train_options)
        self.assertLess(loss, 1)
