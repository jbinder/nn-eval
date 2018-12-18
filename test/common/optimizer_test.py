import logging
import unittest
from unittest.mock import Mock

from common.optimizer import Optimizer
from common.options import OptimizerOptions, TrainOptions


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
