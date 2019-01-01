import abc
from typing import Any

from common.options import NetworkOptions, TrainOptions


class Network(abc.ABC):

    @abc.abstractmethod
    def init(self, data: dict, network_options: NetworkOptions, train_options: TrainOptions) -> None:
        pass

    @abc.abstractmethod
    def train(self) -> TrainOptions:
        """ :returns: the actual options used to train the network """
        pass

    @abc.abstractmethod
    def validate(self) -> float:
        pass

    @abc.abstractmethod
    def predict(self, data) -> Any:
        pass

    @abc.abstractmethod
    def save(self, path: str) -> None:
        pass

    @abc.abstractmethod
    def load(self, path: str) -> '__class__':
        pass
