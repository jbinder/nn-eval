import unittest
from random import uniform

import numpy as np

from common.visualizer import Visualizer


# noinspection PyMethodMayBeStatic
class VisualizerTest(unittest.TestCase):
    def test_plot_valid_input(self):
        x = [[x] for x in np.linspace(0, 100, 10)]
        y = [x[0] for x in x]
        predicted = [x[0] + uniform(-1, 1) for x in x]
        visualizer = Visualizer()
        visualizer.plot(x, y, predicted)
