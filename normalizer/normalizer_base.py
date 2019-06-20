from abc import abstractmethod, ABC

import numpy as np


class NormalizerBase(ABC):

    @staticmethod
    @abstractmethod
    def normalize(data):
        NormalizerBase._assert_is_numpy_array(data)
        return data

    @staticmethod
    @abstractmethod
    def denormalize(data):
        NormalizerBase._assert_is_numpy_array(data)
        return data

    @staticmethod
    def _assert_is_numpy_array(data):
        if type(data) is not np.ndarray:
            raise ValueError("The input data needs to be a numpy array.")
