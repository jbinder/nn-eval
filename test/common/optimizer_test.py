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
        network_mock.validate.return_value = 1
        optimizer = Optimizer()
        optimizer.run(network_mock, {}, None, TrainOptions(None, None, None, "SGD", "MSELoss"), OptimizerOptions(None))
        network_mock.init.assert_called()
        self.assertEqual(optimizer.num_runs_per_setting, network_mock.init.call_count)

    def test_run_no_loss_function_specified_should_enumerate_over_all_loss_functions(self):
        network_mock = Mock()
        network_mock.validate.return_value = 1
        optimizer = Optimizer()
        optimizer.run(network_mock, {}, None, TrainOptions(None, None, None, "SGD", None), OptimizerOptions(None))
        network_mock.init.assert_called()
        self.assertEqual(optimizer.num_runs_per_setting * len(optimizer.loss_functions), network_mock.init.call_count)

    def test_run_no_optimizer_specified_should_enumerate_over_all_optimizers(self):
        network_mock = Mock()
        network_mock.validate.return_value = 1
        optimizer = Optimizer()
        optimizer.run(network_mock, {}, None, TrainOptions(None, None, None, None, "MSELoss"), OptimizerOptions(None))
        network_mock.init.assert_called()
        self.assertEqual(optimizer.num_runs_per_setting * len(optimizer.optimizers), network_mock.init.call_count)

    def test_run_no_optional_options_specified_using_linear_data(self):
        data = self._get_data_linear()
        train_options = TrainOptions(1, 100, True, None, None)
        network_options = NetworkOptions(len(data['train'][0][0]), len(data['train'][1][0]), [8])
        network = PytorchNetwork()
        optimizer = Optimizer()
        best = optimizer.run(network, data, network_options, train_options, OptimizerOptions(None))
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
