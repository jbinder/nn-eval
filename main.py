import argparse
import logging
import numpy as np

from common.csv_data_provider import CsvDataProvider
from common.optimizer import Optimizer
from common.options import TrainOptions, NetworkOptions, OptimizerOptions
from common.visualizer import Visualizer
from networks.pytorch.pytorch_network import PytorchNetwork


def main():
    logging.basicConfig(level=0)

    args = get_parser().parse_args()

    data_provider = CsvDataProvider()
    data = data_provider.get_data_from_file(args.x, args.y, args.data_train_percentage)

    train_options = TrainOptions(args.epochs, args.batch_size, args.print_every, args.gpu, args.optimizer,
                                 args.activation_function, args.loss_function, args.num_runs_per_setting,
                                 args.dropout_rate, args.bias)

    num_features_in = data['train'][0].shape[1]
    num_features_out = data['train'][1].shape[1]
    network_options = NetworkOptions(num_features_in, num_features_out, args.size_hidden)

    network = PytorchNetwork()
    optimizer = Optimizer()
    result = optimizer.run(network, data, network_options, train_options, OptimizerOptions(args.model_file))
    logging.info(f"Minimum loss: {result['loss']} (details: {result})")

    if args.visualize:
        visualizer = Visualizer()
        x = np.concatenate((data['train'][0], data['valid'][0]), 0)
        y = np.concatenate((data['train'][1], data['valid'][1]), 0)
        predicted = network.predict(x)
        visualizer.plot(x, y, predicted)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Tries finding the best training and network options for a specified dataset.',
    )
    parser.add_argument('--x', action="store", default="x.csv")
    parser.add_argument('--y', action="store", default="y.csv")
    parser.add_argument('--data_train_percentage', action="store", type=float, default=0.7)
    parser.add_argument('--size_hidden', nargs="+", type=int, default=None)
    parser.add_argument('--gpu', action="store", type=bool, default=True)
    parser.add_argument('--optimizer', action="store", default=None)
    parser.add_argument('--activation_function', action="store", default="relu")
    parser.add_argument('--loss_function', action="store", default=None)
    parser.add_argument('--dropout_rate', action="store", type=float, default=0.5)
    parser.add_argument('--bias', action="store", type=bool, default=True)
    parser.add_argument('--epochs', action="store", type=int, default=None)
    parser.add_argument('--print_every', action="store", type=int, default=64)
    parser.add_argument('--model_file', action="store", default=None)
    parser.add_argument('--batch_size', action="store", type=int, default=None)
    parser.add_argument('--num_runs_per_setting', action="store", type=int, default=10)
    parser.add_argument('--visualize', action="store", type=bool, default=True)
    return parser


if __name__ == "__main__":
    main()
