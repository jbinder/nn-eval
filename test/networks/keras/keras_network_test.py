import logging
import unittest

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

    @staticmethod
    def _get_data_linear():
        return {
            'train': (
                [[1], [2], [3], [4], [5], [6], [7], [8], [9]],
                [[3], [6], [9], [12], [15], [18], [21], [24], [27]]),
            'valid': ([[10], [11], [12]], [[30], [33], [36]])}

    @staticmethod
    def _get_train_options_linear():
        return TrainOptions(activation_function="none", bias=False, seed=42, deterministic=True)

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

