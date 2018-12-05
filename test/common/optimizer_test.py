import unittest
from unittest.mock import Mock

from common.optimizer import Optimizer
from common.options import OptimizerOptions


class OptimizerTest(unittest.TestCase):
    def test_run_should_call_network_multiple_times(self):
        network_mock = Mock()
        network_mock.validate.return_value = 1
        optimizer = Optimizer()
        optimizer.run(network_mock, {}, None, None, OptimizerOptions(None))
        network_mock.init.assert_called()
        self.assertEqual(10, network_mock.init.call_count)
