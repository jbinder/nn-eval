import unittest
import main
from components.pytorch.PyTorchNetwork import PyTorchNetwork


class PtbTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_run_linear(self):
        data = {
            'train': (
                [[1], [2], [3], [4], [5], [6], [7], [8], [9]],
                [[3], [6], [9], [12], [15], [18], [21], [24], [27]]),
            'valid': ([[10], [11], [12]], [[30], [33], [36]])}
        train_options = main.TrainOptions(50, 100, True)
        network_options = main.NetworkOptions(1, 1, [4, 8])
        network = PyTorchNetwork()
        loss = network.run(data, network_options, train_options)
        self.assertLess(loss, 2)
