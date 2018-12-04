import logging

from common.options import NetworkOptions, TrainOptions, OptimizerOptions
from components.network import Network


class Optimizer:
    def run(self, network: Network, data: dict, network_options: NetworkOptions, train_options: TrainOptions,
            optimizer_options: OptimizerOptions):
        best = {'loss': None}
        for i in range(1, 11):
            logging.info(f"Run #{i}...")
            loss, parameters = network.run(data, network_options, train_options)
            if best['loss'] is None or loss < best['loss']:
                best['loss'] = loss
                if optimizer_options.save_path is not None:
                    network.save(optimizer_options.save_path)
        return best