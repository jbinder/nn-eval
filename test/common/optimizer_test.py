import unittest
from unittest.mock import Mock

from common.optimizer import Optimizer


class OptimizerTest(unittest.TestCase):
    def test_run_should_call_network_multiple_times(self):
        network_mock = Mock()
        network_mock.run.return_value = {'loss': 1, 'seeds': {}}
        optimizer = Optimizer()
        optimizer.run(network_mock, {}, None, None)
        network_mock.run.assert_called()
        self.assertEqual(10, network_mock.run.call_count)
