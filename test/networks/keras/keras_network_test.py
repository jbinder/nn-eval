import logging
import os
import sys
import unittest
from random import shuffle

from numpy.ma import arange

from common.options import TrainOptions, NetworkOptions
from networks.keras.keras_network import KerasNetwork


class KerasNetworkTest(unittest.TestCase):

    epsilon: float

    def __init__(self, *args, **kwargs):
        super(__class__, self).__init__(*args, **kwargs)
        logging.basicConfig(level=0)
        self.epsilon = 0.0001

    def setUp(self):
        pass

    def test_run_linear(self):
        network = self._get_trained_network(self._get_data_linear(), [], self._get_train_options_linear())
        loss = network.validate()
        self.assertLess(loss, self.epsilon)

    def test_run_linear_using_linear_model_predicts_trained_data_accurately(self):
        data = self._get_data_linear()
        network = self._get_trained_network(data, [], self._get_train_options_linear())
        predicted = network.predict(data['train'][0][0])
        y = data['train'][1][0]
        self.assertLess(abs(predicted - y), self.epsilon)

    def test_run_linear_stops_if_done_learning(self):
        train_options = self._get_train_options_linear()._replace(num_epochs=sys.maxsize)
        network = self._get_trained_network(self._get_data_linear(), [], train_options)
        best = network.train()
        self.assertLess(best.num_epochs, sys.maxsize)

    def test_run_linear_2_vars(self):
        train_options = self._get_train_options_linear()._replace(num_epochs=50000, optimizer="adam")
        network = self._get_trained_network(self._get_data_linear_2_vars(), [], train_options)
        loss = network.validate()
        self.assertLess(loss, 0.08)

    def test_run_quadratic(self):
        options = TrainOptions(num_epochs=50000, optimizer="sgd", loss_function="mse", activation_function="sigmoid",
                               deterministic=True, seed=1073676287, batch_size=None, progress_detection_patience=25000)
        network = self._get_trained_network(self._get_data_quadratic(), [100], options)
        loss = network.validate()
        self.assertLess(loss, 300)

    def test_save(self):
        network = self._get_trained_network(self._get_data_linear(), [], self._get_train_options_linear())
        file_name = "tmp.pth"
        network.save(file_name)
        self.assertTrue(os.path.isfile(file_name))
        os.remove(file_name)

    def test_load(self):
        network = self._get_trained_network(self._get_data_linear(), [], self._get_train_options_linear())
        file_name = "tmp.pth"
        network.save(file_name)

        data = self._get_data_linear()
        train_options = TrainOptions()
        network_options = NetworkOptions(1, 1, [])
        network = KerasNetwork()
        network.init(data, network_options, train_options)

        network.load(file_name)
        loss = network.validate()

        os.remove(file_name)
        self.assertLess(loss, self.epsilon)

    def test_predict(self):
        network = self._get_trained_network(self._get_data_linear(), [], self._get_train_options_linear())
        actual = network.predict([2]).item()
        self.assertAlmostEqual(6, actual, 3)

    def test_predict_after_load(self):
        network = self._get_trained_network(self._get_data_linear(), [], self._get_train_options_linear())
        file_name = "tmp.pth"
        network.save(file_name)

        network = KerasNetwork()
        network.load(file_name)

        actual = network.predict([2.0]).item()

        os.remove(file_name)
        self.assertAlmostEqual(6, actual, 3)

    @staticmethod
    def _get_data_linear():
        return {
            'train': (
                [[1], [2], [3], [4], [5], [6], [7], [8], [9]],
                [[3], [6], [9], [12], [15], [18], [21], [24], [27]]),
            'valid': ([[10], [11], [12]], [[30], [33], [36]])}

    @staticmethod
    def _get_data_linear_2_vars():
        train_x = []
        for i in range(1, 101, 1):
            for j in range(101, 1, -1):
                train_x.append([i, j])
        train_y = [[i[0] + i[1]] for i in train_x]
        return {
            'train': (
                train_x,
                train_y),
            'valid': ([[10, 11], [11, 20], [12, 4]], [[21], [31], [16]])}

    @staticmethod
    def _get_data_quadratic():
        train_x = [[i] for i in arange(0.0, 1.0, 0.0001)]  # 101
        shuffle(train_x)
        train_y = [[i[0] * i[0]] for i in train_x]
        return {
            'train': (
                train_x,
                train_y),
            'valid': ([[3.3], [0.2], [0.23]], [[10.89], [0.04], [0.0529]])}

    @staticmethod
    def _get_train_options_linear():
        return TrainOptions(activation_function="linear", bias=False, seed=42, deterministic=False)

    @staticmethod
    def _get_trained_network(data, hidden_layer_sizes, train_options=TrainOptions()):
        default_train_options = TrainOptions(num_epochs=20000, optimizer="adam", loss_function="mse", print_every=100,
                                             use_gpu=True, activation_function="linear", bias=True, dropout_rate=0.5)
        option_dict = {k: v for k, v in train_options._asdict().items() if v is not None}
        option_dict.update({k: None for k, v in train_options._asdict().items() if v == "none"})
        train_options = default_train_options._replace(**option_dict)
        network_options = NetworkOptions(len(data['train'][0][0]), len(data['train'][1][0]), hidden_layer_sizes)
        network = KerasNetwork()
        network.init(data, network_options, train_options)
        network.train()
        return network
