import abc

from common.options import NetworkOptions, TrainOptions


class Network(abc.ABC):

    @abc.abstractmethod
    # def run(self, company_name: str, id: int) -> Employee
    def init(self, data: dict, network_options: NetworkOptions, train_options: TrainOptions) -> None:
        pass

    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def validate(self) -> float:
        pass

    @abc.abstractmethod
    def save(self, path: str) -> None:
        pass

    @abc.abstractmethod
    def load(self, path: str) -> '__class__':
        pass
