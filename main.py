import argparse
import logging
import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd

from common.csv_data_provider import CsvDataProvider
from common.optimizer import Optimizer
from common.options import TrainOptions, NetworkOptions, OptimizerOptions
from common.visualizer import Visualizer
from networks.keras.keras_network import KerasNetwork
from networks.pytorch.pytorch_network import PytorchNetwork
from normalizer.tools import get_normalizer


def main():
    logging.basicConfig(
        level=logging.NOTSET,
        format='%(asctime)s %(levelname)s %(module)s:%(funcName)s: %(message)s',
    )
    args = get_parser().parse_args()

    data_provider = CsvDataProvider()
    data = data_provider.get_data_from_file(args.x, args.y, args.data_train_percentage)

    normalizer = get_normalizer(args.normalizer, np.concatenate((data['train'][0], data['valid'][0]), 0))
    networks = get_networks(args)
    result = None

    if args.mode == "train":
        logging.info("Training...")
        start = time.time()
        result = train(args, data, normalizer, networks)
        elapsed = (time.time() - start)
        logging.info(f"Done: min.loss={result['loss']} time={timedelta(seconds=elapsed)} (details={result})")

    if args.visualize or args.mode == "predict":
        predict(args, data, networks, normalizer, result, args.x_predict)


def predict(args, data, networks, normalizer, result, x_predict):
    if args.mode == "predict":
        if len(networks) > 1:
            raise Exception('In predict mode one single network needs to be set.')
        network = networks[0]
        network.load(args.model_file)
    else:
        network = next((x for x in networks if x.__class__.__name__ == result['network']), None)
    if x_predict is None:
        x = data['valid'][0] if not args.visualize_include_test_data \
            else np.concatenate((data['train'][0], data['valid'][0]), 0)
        y = data['valid'][1] if not args.visualize_include_test_data \
            else np.concatenate((data['train'][1], data['valid'][1]), 0)
    else:
        base_dir = os.getcwd()
        x = pd.read_csv(os.path.join(base_dir, x_predict))
        y = np.array([[0] for i in range(0, x.shape[0])])
    predicted = network.predict(normalizer.normalize(x))
    if y.shape[1] != 1 or predicted.shape[1] != 1:
        raise Exception('Only one-dimensional output variables are currently supported.')
    x_num = [i for i in range(x.shape[0])]
    y_array = [y[0] for y in y]
    predicted_array = [p[0] for p in predicted]
    logging.info("Predicted: " + ", ".join(["{0:0.4f}".format(i) for i in predicted_array]))
    visualize_limit = x.shape[0] if args.visualize_limit is None else args.visualize_limit
    visualizer = Visualizer()
    visualizer.plot(x_num[:visualize_limit], y_array[:visualize_limit], predicted_array[:visualize_limit])


def train(args, data, normalizer, networks):
    normalized_data = {'train': (normalizer.normalize(data['train'][0]), data['train'][1]),
                       'valid': (normalizer.normalize(data['valid'][0]), data['valid'][1])}
    train_options = TrainOptions(args.epochs, args.batch_size, args.print_every, args.gpu, args.optimizer,
                                 args.learning_rate, args.activation_function, args.loss_function,
                                 args.num_runs_per_setting, args.dropout_rate, args.bias, args.seed, args.deterministic,
                                 args.progress_detection_patience, args.progress_detection_min_delta)
    num_features_in = data['train'][0].shape[1]
    num_features_out = data['train'][1].shape[1]
    network_options = NetworkOptions(num_features_in, num_features_out, args.size_hidden)
    optimizer = Optimizer()
    result = optimizer.run(networks, normalized_data, network_options, train_options, OptimizerOptions(args.model_file))
    return result


def get_networks(args):
    networks = []
    if args.networks is None or "pytorch" in args.networks:
        networks.append(PytorchNetwork())
    if args.networks is None or "keras" in args.networks:
        networks.append(KerasNetwork())
    return networks


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
    parser.add_argument('--learning_rate', action="store", type=float, default=0.001)
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
    parser.add_argument('--progress_detection_patience', action="store", type=int, default=None)
    parser.add_argument('--progress_detection_min_delta', action="store", type=float, default=0)
    parser.add_argument('--normalizer', action="store", default="Identity")
    parser.add_argument('--mode', action="store", default="train")
    parser.add_argument('--x_predict', action="store", default="x_predict.csv")
    return parser


if __name__ == "__main__":
    # fix logger collision, see https://github.com/tensorflow/tensorflow/issues/26691
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    main()
