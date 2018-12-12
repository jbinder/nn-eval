import logging
import os
import unittest

from common.options import TrainOptions, NetworkOptions
from networks.pytorch.pytorch_network import PytorchNetwork


class PyTorchNetworkTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(__class__, self).__init__(*args, **kwargs)
        logging.basicConfig(level=0)

    def setUp(self):
        pass

    def test_run_linear(self):
        network = self._get_trained_network(self._get_data_linear(), self._get_hidden_layer_sizes_linear())
        loss = network.validate()
        self.assertLess(loss, 2)

    def test_run_linear_2_vars(self):
        network = self._get_trained_network(self._get_data_linear_2_vars(), [64, 64])
        loss = network.validate()
        self.assertLess(loss, 2)

    def test_save(self):
        network = self._get_trained_network(self._get_data_linear(), self._get_hidden_layer_sizes_linear())
        file_name = "tmp.pth"
        network.save(file_name)
        self.assertTrue(os.path.isfile(file_name))
        os.remove(file_name)

    def test_load(self):
        network = self._get_trained_network(self._get_data_linear(), self._get_hidden_layer_sizes_linear())
        file_name = "tmp.pth"
        network.save(file_name)

        data = self._get_data_linear()
        train_options = TrainOptions(50, 100, True)
        network_options = NetworkOptions(1, 1, [4, 8])
        network = PytorchNetwork()
        network.init(data, network_options, train_options)

        network.load(file_name)

        loss = network.validate()
        self.assertLess(loss, 2)
        os.remove(file_name)

    def test_predict(self):
        network = self._get_trained_network(self._get_data_linear(), self._get_hidden_layer_sizes_linear())
        actual = network.predict([2]).item()
        self.assertAlmostEqual(6, actual, 0)

    def test_predict_after_load(self):
        network = self._get_trained_network(self._get_data_linear(), self._get_hidden_layer_sizes_linear())
        file_name = "tmp.pth"
        network.save(file_name)

        network = PytorchNetwork()
        network.load(file_name)

        actual = network.predict([2.0]).item()

        self.assertAlmostEqual(6, actual, 0)
        os.remove(file_name)

    @staticmethod
    def _get_data_linear():
        return {
            'train': (
                [[1], [2], [3], [4], [5], [6], [7], [8], [9]],
                [[3], [6], [9], [12], [15], [18], [21], [24], [27]]),
            'valid': ([[10], [11], [12]], [[30], [33], [36]])}

    @staticmethod
    def _get_data_linear_2_vars():
        train_x = [[i, i+1] for i in range(1, 101, 1)]
        train_y = [[i[0] + i[1]] for i in train_x]
        return {
            'train': (
                train_x,
                train_y),
            'valid': ([[10, 11], [11, 20], [12, 4]], [[21], [31], [16]])}

    @staticmethod
    def _get_hidden_layer_sizes_linear():
        return [4, 8]

    def _get_trained_network(self, data, hidden_layer_sizes):
        train_options = TrainOptions(500, 100, True)
        network_options = NetworkOptions(len(data['train'][0][0]), len(data['train'][1][0]), hidden_layer_sizes)
        network = PytorchNetwork()
        network.init(data, network_options, train_options)
        network.train()
        return network
