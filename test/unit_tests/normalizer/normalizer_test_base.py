import abc
import unittest

import numpy as np


class NormalizerTestBase(unittest.TestCase, abc.ABC):
    def __init__(self, *args, **kwargs):
        super(__class__, self).__init__(*args, **kwargs)
        self.source = np.array([0, 0.5, 0.75, 1, 0, 11])