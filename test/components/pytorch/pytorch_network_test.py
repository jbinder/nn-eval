import os
import unittest

from common.options import TrainOptions, NetworkOptions
from components.pytorch.pytorch_network import PytorchNetwork


class PyTorchNetworkTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_run_linear(self):
        network = self._get_trained_network()
        loss = network.validate()
        self.assertLess(loss, 2)

    def test_save(self):
        network = self._get_trained_network()
        file_name = "tmp.pth"
        network.save(file_name)
        self.assertTrue(os.path.isfile(file_name))
        os.remove(file_name)

    def test_load(self):
        network = self._get_trained_network()
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

    def test_predict(self):
        network = self._get_trained_network()
        actual = network.predict([2])
        self.assertAlmostEqual(6, actual, 0)

    @staticmethod
    def _get_data_linear():
        return {
            'train': (
                [[1], [2], [3], [4], [5], [6], [7], [8], [9]],
                [[3], [6], [9], [12], [15], [18], [21], [24], [27]]),
            'valid': ([[10], [11], [12]], [[30], [33], [36]])}

    def _get_trained_network(self):
        data = self._get_data_linear()
        train_options = TrainOptions(500, 100, True)
        network_options = NetworkOptions(1, 1, [4, 8])
        network = PytorchNetwork()
        network.init(data, network_options, train_options)
        network.train()
        return network
