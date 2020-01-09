import logging
import unittest
from unittest.mock import Mock

from common.optimizer import Optimizer
from common.options import OptimizerOptions, TrainOptions, NetworkOptions
from networks.pytorch.pytorch_network import PytorchNetwork


class OptimizerTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(__class__, self).__init__(*args, **kwargs)
        logging.basicConfig(level=0)

    def test_run_should_call_network_multiple_times(self):
        network_mock = Mock()
        network_mock.__class__.__name__ = "PytorchNetwork"
        network_mock.validate.return_value = 1
        optimizer = Optimizer()
        optimizer.run([network_mock], {}, NetworkOptions(0, 0, []),
                      TrainOptions(optimizer="adam", loss_function="mse"), OptimizerOptions(None))
        network_mock.init.assert_called()
        self.assertEqual(optimizer.default_num_runs_per_setting, network_mock.init.call_count)

    def test_run_should_call_network_specified_times(self):
        network_mock = Mock()
        network_mock.__class__.__name__ = "PytorchNetwork"
        network_mock.validate.return_value = 1
        optimizer = Optimizer()
        num_runs = 100
        optimizer.run([network_mock], {}, NetworkOptions(0, 0, []),
                      TrainOptions(optimizer="adam", loss_function="mse", num_runs_per_setting=num_runs),
                      OptimizerOptions(None))
        network_mock.init.assert_called()
        self.assertEqual(num_runs, network_mock.init.call_count)

    def test_run_no_loss_function_specified_should_enumerate_over_all_loss_functions(self):
        network_mock = Mock()
        network_mock.__class__.__name__ = "PytorchNetwork"
        network_mock.validate.return_value = 1
        optimizer = Optimizer()
        optimizer.run([network_mock], {}, NetworkOptions(0, 0, []),
                      TrainOptions(optimizer="adam"), OptimizerOptions(None))
        network_mock.init.assert_called()
        self.assertEqual(optimizer.default_num_runs_per_setting * len(optimizer.loss_functions),
                         network_mock.init.call_count)

    def test_run_no_optimizer_specified_should_enumerate_over_all_optimizers(self):
        network_mock = Mock()
        network_mock.__class__.__name__ = "PytorchNetwork"
        network_mock.validate.return_value = 1
        optimizer = Optimizer()
        optimizer.run([network_mock], {}, NetworkOptions(0, 0, []),
                      TrainOptions(loss_function="mse"), OptimizerOptions(None))
        network_mock.init.assert_called()
        self.assertEqual(optimizer.default_num_runs_per_setting * len(optimizer.optimizers), network_mock.init.call_count)

    def test_run_no_hidden_layers_specified_should_enumerate_over_all_hidden_layers(self):
        network_mock = Mock()
        network_mock.__class__.__name__ = "PytorchNetwork"
        network_mock.validate.return_value = 1
        optimizer = Optimizer()
        optimizer.run([network_mock], {}, NetworkOptions(0, 0, None),
                      TrainOptions(optimizer="adam", loss_function="mse"), OptimizerOptions(None))
        network_mock.init.assert_called()
        self.assertEqual(optimizer.default_num_runs_per_setting * len(optimizer.hidden_layers), network_mock.init.call_count)

    def test_run_no_optional_options_specified_using_linear_data(self):
        data = self._get_data_linear()
        train_options = TrainOptions(print_every=100, use_gpu=True, activation_function="relu", num_epochs=100)
        network_options = NetworkOptions(len(data['train'][0][0]), len(data['train'][1][0]), None)
        network = PytorchNetwork()
        optimizer = Optimizer()
        best = optimizer.run([network], data, network_options, train_options, OptimizerOptions(None))
        logging.info(f"Best run: {best}")
        self.assertLess(best['loss'], 100)
        self.assertIsNotNone(best['train_options'])

    @staticmethod
    def _get_data_linear():
        return {
            'train': (
                [[1], [2], [3], [4], [5], [6], [7], [8], [9]],
                [[3], [6], [9], [12], [15], [18], [21], [24], [27]]),
            'valid': ([[10], [11], [12]], [[30], [33], [36]])}
