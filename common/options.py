from collections import namedtuple

TrainOptions = namedtuple('TrainOptions',
                          ['num_epochs', 'batch_size', 'print_every', 'use_gpu', 'optimizer', 'activation_function',
                           'loss_function', 'num_runs_per_setting', 'dropout_rate', 'bias', 'seed', 'deterministic'])
TrainOptions.__new__.__defaults__ = (None,) * len(TrainOptions._fields)
NetworkOptions = namedtuple('NetworkOptions', ['input_layer_size', 'output_layer_size', 'hidden_layer_sizes'])
NetworkOptions.__new__.__defaults__ = (None,) * len(NetworkOptions._fields)
OptimizerOptions = namedtuple('OptimizerOptions', ['save_path'])
OptimizerOptions.__new__.__defaults__ = (None,) * len(OptimizerOptions._fields)
