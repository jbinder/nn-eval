import argparse
import logging

from common.csv_data_provider import CsvDataProvider
from common.optimizer import Optimizer
from common.options import TrainOptions, NetworkOptions, OptimizerOptions
from networks.pytorch.pytorch_network import PytorchNetwork


def main():
    logging.basicConfig(level=0)

    args = get_parser().parse_args()

    data_provider = CsvDataProvider()
    data = {'train': data_provider.get_data_from_file(args.x, args.y),
            'valid': data_provider.get_data_from_file(args.xvalid, args.yvalid)}

    train_options = TrainOptions(args.epochs, args.print_every, args.gpu, args.optimizer, args.loss_function)

    num_features_in = data['train'][0].shape[1]
    num_features_out = data['train'][1].shape[1]
    network_options = NetworkOptions(num_features_in, num_features_out, args.size_hidden)

    network = PytorchNetwork()
    optimizer = Optimizer()
    result = optimizer.run(network, data, network_options, train_options, OptimizerOptions(args.model_file))
    logging.info(f"Minimum loss: {result['loss']} (details: {result})")


def get_parser():
    parser = argparse.ArgumentParser(
        description='TODO: ',
    )
    parser.add_argument('--x', action="store", default="x.csv")
    parser.add_argument('--y', action="store", default="y.csv")
    parser.add_argument('--xvalid', action="store", default="x_valid.csv")
    parser.add_argument('--yvalid', action="store", default="y_valid.csv")
    parser.add_argument('--size_hidden', nargs="+", type=int, default=[4, 8])
    parser.add_argument('--gpu', action="store", type=bool, default=True)
    parser.add_argument('--optimizer', action="store", default="SGD")
    parser.add_argument('--loss_function', action="store", default="MSELoss")
    parser.add_argument('--epochs', action="store", type=int, default=300)
    parser.add_argument('--print_every', action="store", type=int, default=64)
    parser.add_argument('--model_file', action="store", default=None)
    # parser.add_argument('--batch_size', action="store", type=int, default=64)
    return parser


if __name__ == "__main__":
    main()
