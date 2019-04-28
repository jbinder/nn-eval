import argparse
import logging
import numpy as np

from common.csv_data_provider import CsvDataProvider
from common.optimizer import Optimizer
from common.options import TrainOptions, NetworkOptions, OptimizerOptions
from common.visualizer import Visualizer
from networks.keras.keras_network import KerasNetwork
from networks.pytorch.pytorch_network import PytorchNetwork


def main():
    logging.basicConfig(level=0)

    args = get_parser().parse_args()

    data_provider = CsvDataProvider()
    data = data_provider.get_data_from_file(args.x, args.y, args.data_train_percentage)

    train_options = TrainOptions(args.epochs, args.batch_size, args.print_every, args.gpu, args.optimizer,
                                 args.activation_function, args.loss_function, args.num_runs_per_setting,
                                 args.dropout_rate, args.bias, args.seed, args.deterministic)

    num_features_in = data['train'][0].shape[1]
    num_features_out = data['train'][1].shape[1]
    network_options = NetworkOptions(num_features_in, num_features_out, args.size_hidden)

    networks = []
    if args.networks is None or "pytorch" in args.networks:
        networks.append(PytorchNetwork())
    if args.networks is None or "keras" in args.networks:
        networks.append(KerasNetwork())
    optimizer = Optimizer()
    result = optimizer.run(networks, data, network_options, train_options, OptimizerOptions(args.model_file))
    logging.info(f"Minimum loss: {result['loss']} (details: {result})")

    if args.visualize:
        network = next((x for x in networks if x.__class__.__name__ == result['network']), None)
        visualizer = Visualizer()
        x = np.concatenate((data['train'][0], data['valid'][0]), 0)
        y = np.concatenate((data['train'][1], data['valid'][1]), 0)
        predicted = network.predict(x)
        if y.shape[1] != 1 or predicted.shape[1] != 1:
            raise Exception('Only one-dimensional output variables are currently supported.')
        x_num = [i for i in range(x.shape[0])]
        y_array = [y[0] for y in y]
        predicted_array = [p[0] for p in predicted]
        visualizer.plot(x_num, y_array, predicted_array)


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
    parser.add_argument('--seed', action="store", type=int, default=None)
    parser.add_argument('--deterministic', action="store", type=bool, default=False)
    parser.add_argument('--num_runs_per_setting', action="store", type=int, default=10)
    parser.add_argument('--visualize', action="store", type=bool, default=True)
    parser.add_argument('--networks', nargs="+", action="store", default=None)
    return parser


if __name__ == "__main__":
    main()
