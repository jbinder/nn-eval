import unittest

from common.options import TrainOptions, NetworkOptions
from components.pytorch.pytorch_network import PytorchNetwork


class PyTorchNetworkTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_run_linear(self):
        data = {
            'train': (
                [[1], [2], [3], [4], [5], [6], [7], [8], [9]],
                [[3], [6], [9], [12], [15], [18], [21], [24], [27]]),
            'valid': ([[10], [11], [12]], [[30], [33], [36]])}
        train_options = TrainOptions(50, 100, True)
        network_options = NetworkOptions(1, 1, [4, 8])
        network = PytorchNetwork()
        result = network.run(data, network_options, train_options)
        self.assertLess(result[0], 2)
