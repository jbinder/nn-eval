import array
import logging

from common.options import NetworkOptions, TrainOptions, OptimizerOptions
from networks.network import Network


class Optimizer:

    loss_functions: array
    num_runs_per_setting: int

    def __init__(self):
        self.loss_functions = ["L1Loss", "MSELoss", "CrossEntropyLoss"]
        self.num_runs_per_setting = 10

    def run(self, network: Network, data: dict, network_options: NetworkOptions, train_options: TrainOptions,
            optimizer_options: OptimizerOptions):
        best = {'loss': None}
        loss_functions = self.loss_functions if train_options.loss_function is None else [train_options.loss_function]
        for loss_function in loss_functions:
            for i in range(1, self.num_runs_per_setting + 1):
                current_train_options = TrainOptions(
                    train_options.num_epochs,
                    train_options.print_every,
                    train_options.use_gpu,
                    train_options.optimizer,
                    loss_function)
                logging.info(f"Run #{i}: {current_train_options}...")
                network.init(data, network_options, current_train_options)
                network.train()
                loss = network.validate()
                if best['loss'] is None or loss < best['loss']:
                    logging.info("New best run!")
                    best['loss'] = loss
                    if optimizer_options.save_path is not None:
                        network.save(optimizer_options.save_path)
        return best
