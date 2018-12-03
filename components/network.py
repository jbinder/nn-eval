import abc

from common.options import NetworkOptions, TrainOptions


class Network(abc.ABC):

    @abc.abstractmethod
    # def run(self, company_name: str, id: int) -> Employee
    def run(self, data: dict, network_options: NetworkOptions, train_options: TrainOptions) -> (int, dict):
        pass
