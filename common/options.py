from collections import namedtuple

TrainOptions = namedtuple('TrainOptions',
                          ['num_epochs', 'batch_size', 'print_every', 'use_gpu', 'optimizer', 'loss_function',
                           'num_runs_per_setting'])
NetworkOptions = namedtuple('NetworkOptions', ['input_layer_size', 'output_layer_size', 'hidden_layer_sizes'])
OptimizerOptions = namedtuple('OptimizerOptions', ['save_path'])
