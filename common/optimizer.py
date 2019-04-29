import array
import logging
from typing import List

from common.options import NetworkOptions, TrainOptions, OptimizerOptions
from networks.network import Network


class Optimizer:

    num_runs_per_setting: int
    loss_functions: array
    loss_functions_keras: array
    loss_functions_pytorch: array
    optimizers: array

    def __init__(self):
        self.default_num_runs_per_setting = 10
        self.loss_functions = ["mse"]  # TODO: "PyTorch: CrossEntropyLoss, L1Loss",
        self.loss_functions_map = {
            "KerasNetwork": {"mse": "mean_squared_error"},
            "PytorchNetwork": {"mse": "MSELoss"},
        }
        self.optimizers = ["SGD", "adam"]
        self.optimizers_map = {
            "KerasNetwork": {"adam": "adam"},
            "PytorchNetwork": {"adam": "Adam"},
        }
        self.hidden_layers = [[8], [64], [512], [2048], [8, 8], [64, 64], [512, 512], [2048, 2048]]

    def run(self, networks: List[Network], data: dict, network_options: NetworkOptions, train_options: TrainOptions,
            optimizer_options: OptimizerOptions):
        best = {'loss': None, 'train_options': None, 'network_options': None, 'network': None}
        for network in networks:
            logging.info(f"Using {network.__class__.__name__}...")
            # for all hidden layers
            all_hidden_layer_sizes = self.hidden_layers if network_options.hidden_layer_sizes is None else \
                [network_options.hidden_layer_sizes]
            for hidden_layer_sizes in all_hidden_layer_sizes:
                network_options = NetworkOptions(network_options.input_layer_size, network_options.output_layer_size,
                                                 hidden_layer_sizes)
                # for all optimizers ...
                optimizers = self.optimizers if train_options.optimizer is None else [train_options.optimizer]
                for optimizer in optimizers:
                    optimizer = self.optimizers_map[network.__class__.__name__][optimizer]
                    options = train_options._replace(optimizer=optimizer)
                    # ... and for all loss functions ...
                    loss_functions = self.loss_functions if train_options.loss_function is None else \
                        [train_options.loss_function]
                    for loss_function in loss_functions:
                        loss_function = self.loss_functions_map[network.__class__.__name__][loss_function]
                        options = options._replace(loss_function=loss_function)
                        # ... run x times (try several times because of the random seed)
                        num_runs = train_options.num_runs_per_setting \
                            if train_options.num_runs_per_setting is not None else self.default_num_runs_per_setting
                        for i in range(1, num_runs + 1):
                            self._run_once(best, options, data, i, network, network_options, optimizer_options)
        return best

    @staticmethod
    def _run_once(best, train_options, data, i, network, network_options, optimizer_options):
        logging.info(f"Run #{i}: {train_options}, {network_options}...")
        network.init(data, network_options, train_options)
        actual_train_options = network.train()
        loss = network.validate()
        logging.info(f"Loss: {loss}")
        if best['loss'] is None or loss < best['loss']:
            logging.info("New best run!")
            best['loss'] = loss
            best['train_options'] = actual_train_options
            best['network_options'] = network_options
            best['network'] = network.__class__.__name__
            if optimizer_options.save_path is not None:
                network.save(optimizer_options.save_path)
