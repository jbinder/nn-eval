import logging

from common.options import NetworkOptions, TrainOptions
from components.network import Network


class Optimizer:
    def run(self, network: Network, data: dict, network_options: NetworkOptions, train_options: TrainOptions):
        best = {'loss': None, 'seeds': None}
        for i in range(1, 11):
            logging.info(f"Run #{i}...")
            loss, seeds = network.run(data, network_options, train_options)
            if best['loss'] is None or loss < best['loss']:
                best['loss'] = loss
                best['seeds'] = seeds
        return best
