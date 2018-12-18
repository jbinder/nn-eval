import array
import logging

from common.options import NetworkOptions, TrainOptions, OptimizerOptions
from networks.network import Network


class Optimizer:

    num_runs_per_setting: int
    loss_functions: array
    optimizers: array

    def __init__(self):
        self.num_runs_per_setting = 10
        self.loss_functions = ["L1Loss", "MSELoss", "CrossEntropyLoss"]
        self.optimizers = ["SGD", "Adam"]

    def run(self, network: Network, data: dict, network_options: NetworkOptions, train_options: TrainOptions,
            optimizer_options: OptimizerOptions):
        best = {'loss': None}
        # for all optimizers ...
        optimizers = self.optimizers if train_options.optimizer is None else [train_options.optimizer]
        for optimizer in optimizers:
            options = TrainOptions(train_options.num_epochs, train_options.print_every,
                                   train_options.use_gpu, optimizer, train_options.loss_function)
            # ... and for all loss functions ...
            loss_functions = self.loss_functions if train_options.loss_function is None else \
                [train_options.loss_function]
            for loss_function in loss_functions:
                options = TrainOptions(options.num_epochs, options.print_every,
                                       options.use_gpu, options.optimizer, loss_function)
                # ... run x times (try several times because of the random seed)
                for i in range(1, self.num_runs_per_setting + 1):
                    self._run_once(best, options, data, i, network, network_options, optimizer_options)
        return best

    @staticmethod
    def _run_once(best, current_train_options, data, i, network, network_options, optimizer_options):
        logging.info(f"Run #{i}: {current_train_options}...")
        network.init(data, network_options, current_train_options)
        network.train()
        loss = network.validate()
        logging.info(f"Loss: {loss}")
        if best['loss'] is None or loss < best['loss']:
            logging.info("New best run!")
            best['loss'] = loss
            if optimizer_options.save_path is not None:
                network.save(optimizer_options.save_path)
