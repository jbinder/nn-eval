from collections import namedtuple

TrainOptions = namedtuple('TrainOptions', ['num_epochs', 'print_every', 'use_gpu'])
NetworkOptions = namedtuple('NetworkOptions', ['input_layer_size', 'output_layer_size', 'hidden_layer_sizes'])
OptimizerOptions = namedtuple('OptimizerOptions', ['save_path'])
