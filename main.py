import argparse
import importlib
import logging
import time
from datetime import timedelta

import numpy as np

from common.csv_data_provider import CsvDataProvider
from common.optimizer import Optimizer
from common.options import TrainOptions, NetworkOptions, OptimizerOptions
from common.visualizer import Visualizer
from networks.keras.keras_network import KerasNetwork
from networks.pytorch.pytorch_network import PytorchNetwork


def main():
    logging.basicConfig(
        level=0,
        format='%(asctime)s %(levelname)s %(module)s:%(funcName)s: %(message)s',
    )
    logging.info("Starting...")
    start = time.time()
    args = get_parser().parse_args()

    data_provider = CsvDataProvider()
    data = data_provider.get_data_from_file(args.x, args.y, args.data_train_percentage)

    normalizer = get_normalizer(args.normalizer)
    normalized_data = {'train': (normalizer.normalize(data['train'][0]), normalizer.normalize(data['train'][1])),
                       'valid': (normalizer.normalize(data['valid'][0]), normalizer.normalize(data['valid'][1]))}

    train_options = TrainOptions(args.epochs, args.batch_size, args.print_every, args.gpu, args.optimizer,
                                 args.activation_function, args.loss_function, args.num_runs_per_setting,
                                 args.dropout_rate, args.bias, args.seed, args.deterministic,
                                 args.progress_detection_patience, args.progress_detection_min_delta)

    num_features_in = data['train'][0].shape[1]
    num_features_out = data['train'][1].shape[1]
    network_options = NetworkOptions(num_features_in, num_features_out, args.size_hidden)

    networks = []
    if args.networks is None or "pytorch" in args.networks:
        networks.append(PytorchNetwork())
    if args.networks is None or "keras" in args.networks:
        networks.append(KerasNetwork())
    optimizer = Optimizer()
    result = optimizer.run(networks, normalized_data, network_options, train_options, OptimizerOptions(args.model_file))
    elapsed = (time.time() - start)
    logging.info(f"Done: min.loss={result['loss']} time={timedelta(seconds=elapsed)} (details={result})")

    if args.visualize:
        network = next((x for x in networks if x.__class__.__name__ == result['network']), None)
        visualizer = Visualizer()
        x = data['valid'][0] if not args.visualize_include_test_data \
            else np.concatenate((data['train'][0], data['valid'][0]), 0)
        y = data['valid'][1] if not args.visualize_include_test_data \
            else np.concatenate((data['train'][1], data['valid'][1]), 0)
        predicted = normalizer.denormalize(network.predict(normalizer.normalize(x)))
        if y.shape[1] != 1 or predicted.shape[1] != 1:
            raise Exception('Only one-dimensional output variables are currently supported.')
        x_num = [i for i in range(x.shape[0])]
        y_array = [y[0] for y in y]
        predicted_array = [p[0] for p in predicted]
        visualize_limit = x.shape[0] if args.visualize_limit is None else args.visualize_limit
        visualizer.plot(x_num[:visualize_limit], y_array[:visualize_limit], predicted_array[:visualize_limit])


def get_normalizer(normalizer):
    try:
        module = importlib.import_module("normalizer." + normalizer.lower() + "_normalizer")
        normalizer = getattr(module, normalizer + "Normalizer")()
        return normalizer
    except ModuleNotFoundError:
        raise Exception(f"Normalizer not supported: {normalizer}")


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
    parser.add_argument('--visualize_limit', action="store", type=int, default=None)
    parser.add_argument('--visualize_include_test_data', action="store", type=bool, default=False)
    parser.add_argument('--networks', nargs="+", action="store", default=None)
    parser.add_argument('--progress_detection_patience', action="store", type=int, default=1000)
    parser.add_argument('--progress_detection_min_delta', action="store", type=float, default=0)
    parser.add_argument('--normalizer', action="store", default="Identity")
    return parser


if __name__ == "__main__":
    main()
