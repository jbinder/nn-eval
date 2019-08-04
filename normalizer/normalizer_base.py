from abc import abstractmethod, ABC

import numpy as np


class NormalizerBase(ABC):

    @abstractmethod
    def normalize(self, data):
        NormalizerBase._assert_is_numpy_array(data)
        return data

    @abstractmethod
    def denormalize(self, data):
        NormalizerBase._assert_is_numpy_array(data)
        return data

    @staticmethod
    def _assert_is_numpy_array(data):
        if type(data) is not np.ndarray:
            raise ValueError("The input data needs to be a numpy array.")
