import argparse
import os
from collections import namedtuple
from math import fabs

import pandas as pd
import torch
import torch.utils.data as utils_data
from torch import nn, optim

from fc_model import Network

TrainOptions = namedtuple('TrainOptions', ['num_epochs', 'print_every', 'use_gpu'])
NetworkOptions = namedtuple('NetworkOptions', ['input_layer_size', 'output_layer_size', 'hidden_layer_sizes'])


def main():
    args = get_parser().parse_args()

    data_loader = {'train': get_data_loader_from_file_name(args.x, args.y),
                   'valid': get_data_loader_from_file_name(args.xvalid, args.yvalid, 1)}

    train_options = TrainOptions(args.epochs, args.print_every, args.gpu)

    num_features_in = data_loader['train'].dataset.tensors[0].shape[1]
    num_features_out = data_loader['train'].dataset.tensors[1].shape[1]
    network_options = NetworkOptions(num_features_in, num_features_out, args.size_hidden)

    run(data_loader, network_options, train_options)


def run(data_loader, network_options, train_options):
    device = get_device(train_options.use_gpu)
    nw = Network(
        network_options.input_layer_size,
        network_options.output_layer_size,
        network_options.hidden_layer_sizes,
        0)
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(nw.parameters(), lr=0.001)
    optimizer = optim.SGD(nw.parameters(), lr=0.001, momentum=0.4)
    nw = nw.to(device)
    train(train_options, criterion, data_loader['train'], device, nw, optimizer)
    loss = validate(data_loader['valid'], device, nw)
    return loss


def validate(data_loader, device, nw):
    print("Validating...")
    loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        actual = predict(device, nw, data)
        current_loss = fabs(target.item() - actual.item())
        loss += current_loss
        print(f"input: {data.item()}, expected: {target.item()}, actual: {actual.item()}, loss: {current_loss}")
    print(f"total loss: {loss}")
    return loss


def get_device(gpu):
    return torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")


def predict(device, nw, data):
    nw.eval()
    with torch.no_grad():
        # noinspection PyCallingNonCallable
        output = nw.forward(torch.tensor(data).to(device).float())
    return output


def train(options, criterion, data_loader, device, nw, optimizer):
    nw.train()
    for epoch in range(options.num_epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            outputs = nw(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % options.print_every == 0:
                print("Epoch: {}/{}.. ".format(epoch + 1, options.num_epochs),
                      "Progress~: {:.2f}.. ".format(
                          ((1 + batch_idx) * len(data)) / (len(data_loader) * len(data)) * 100),
                      "Training Loss: {:.3f}.. ".format(running_loss / options.print_every))
                running_loss = 0.0


def get_data_loader_from_file_name(file_name_x, file_name_y, batch_size=None):
    base_dir = os.getcwd()
    x = pd.read_csv(os.path.join(base_dir, file_name_x))
    y = pd.read_csv(os.path.join(base_dir, file_name_y))
    return get_data_loader(x, y, batch_size)


def get_data_loader(x, y, batch_size=None):
    # noinspection PyUnusedLocal
    common_keys = pd.Index(x.iloc[:, 0]).intersection(pd.Index(y.iloc[:, 0]))
    x_id_col_name = x.axes[1][0]
    x = x.query(f"{x_id_col_name} in @common_keys").iloc[:, 1:].values
    x = torch.FloatTensor(x)
    y_id_col_name = y.axes[1][0]
    y = y.query(f"{y_id_col_name} in @common_keys").iloc[:, 1:].values
    y = torch.FloatTensor(y)
    training_samples = utils_data.TensorDataset(x, y)
    batch_size = batch_size if batch_size is not None else len(x)
    data_loader = utils_data.DataLoader(training_samples, batch_size=batch_size, shuffle=True)
    return data_loader


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
    parser.add_argument('--epochs', action="store", type=int, default=300)
    parser.add_argument('--print_every', action="store", type=int, default=64)
    # parser.add_argument('--batch_size', action="store", type=int, default=64)
    return parser


if __name__ == "__main__":
    main()
