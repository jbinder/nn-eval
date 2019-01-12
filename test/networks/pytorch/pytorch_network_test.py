import logging
import os
import sys
import unittest

from numpy.ma import arange

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

    def test_run_linear_using_linear_model_predicts_trained_data_accurately(self):
        data = self._get_data_linear()
        train_options = TrainOptions(activation_function="none", bias=False)
        network = self._get_trained_network(data, [], train_options)
        predicted = network.predict(data['train'][0][0])
        y = data['train'][1][0]
        epsilon = 0.0001
        self.assertLess(abs(predicted - y), epsilon)
        self.assertLess(abs(network.nw.output.weight.item() - 3), epsilon)

    def test_run_linear_stops_if_done_learning(self):
        network = self._get_trained_network(self._get_data_linear(), self._get_hidden_layer_sizes_linear(),
                                            TrainOptions(num_epochs=sys.maxsize))
        best = network.train()
        self.assertLess(best.batch_size, sys.maxsize)

    def test_run_linear_2_vars(self):
        network = self._get_trained_network(self._get_data_linear_2_vars(), [64, 64],
                                            TrainOptions(num_epochs=1500, optimizer="Adam"))
        loss = network.validate()
        self.assertLess(loss, 5)

    def test_run_quadratic(self):
        network = self._get_trained_network(self._get_data_quadratic(), [4096, 4096],
                                            TrainOptions(num_epochs=10000, optimizer="Adam", loss_function="L1Loss"))
        loss = network.validate()
        self.assertLess(loss, 10)

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
        train_options = TrainOptions(num_epochs=50, print_every=100, use_gpu=True, optimizer="SGD",
                                     loss_function="MSELoss")
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
    def _get_data_quadratic():
        train_x = [[i] for i in arange(0.0, 101.0, 0.25)]
        train_y = [[i[0] * i[0]] for i in train_x]
        return {
            'train': (
                train_x,
                train_y),
            'valid': ([[3.3], [0.2], [44]], [[10.89], [0.04], [1936]])}

    @staticmethod
    def _get_hidden_layer_sizes_linear():
        return [4, 8]

    @staticmethod
    def _get_trained_network(data, hidden_layer_sizes, train_options=TrainOptions()):
        default_train_options = TrainOptions(num_epochs=500, optimizer="SGD", loss_function="MSELoss", print_every=100,
                                             use_gpu=True, activation_function="relu", bias=True, dropout_rate=0.5)
        option_dict = {k: v for k, v in train_options._asdict().items() if v is not None}
        option_dict.update({k: None for k, v in train_options._asdict().items() if v == "none"})
        train_options = default_train_options._replace(**option_dict)
        network_options = NetworkOptions(len(data['train'][0][0]), len(data['train'][1][0]), hidden_layer_sizes)
        network = PytorchNetwork()
        network.init(data, network_options, train_options)
        network.train()
        return network
