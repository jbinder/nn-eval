import argparse
import os

import pandas as pd
import torch
import torch.utils.data as utils_data
from torch import nn, optim

from fc_model import Network


def main():
    args = get_parser().parse_args()

    base_dir = os.getcwd()

    y = pd.read_csv(os.path.join(base_dir, args.y))
    x = pd.read_csv(os.path.join(base_dir, args.x))

    num_features_in = x.shape[1] - 1
    num_features_out = y.shape[1] - 1
    nw = Network(num_features_in, num_features_out, [128], 0)  # TODO: get layers from input

    x, data_loader, y = get_dataloader(x, y)

    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(nw.parameters(), lr=0.001)
    optimizer = optim.Adam(nw.parameters(), lr=0.001)

    device = get_device(args)
    nw = nw.to(device)

    train(args, criterion, data_loader, device, nw, optimizer)
    # TODO: validate


def get_device(args):
    return torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")


# def predict(device, nw):
#     nw.eval()
#     with torch.no_grad():
#         # noinspection PyCallingNonCallable
#         output = nw.forward(torch.tensor(array([TODO])).to(device).float())
#     return output


def train(args, criterion, data_loader, device, nw, optimizer):
    nw.train()
    for epoch in range(args.epochs):
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
            if batch_idx % args.print_every == 0:
                print("Epoch: {}/{}.. ".format(epoch + 1, args.epochs),
                      "Progress~: {:.2f}.. ".format(
                          ((1 + batch_idx) * len(data)) / (len(data_loader) * len(data)) * 100),
                      "Training Loss: {:.3f}.. ".format(running_loss / args.print_every))
                running_loss = 0.0


def get_dataloader(x, y):
    # noinspection PyUnusedLocal
    common_keys = pd.Index(x.iloc[:, 0]).intersection(pd.Index(y.iloc[:, 0]))
    x_id_col_name = x.axes[1][0]
    x = x.query(f"{x_id_col_name} in @common_keys")
    # noinspection PyCallingNonCallable
    x = torch.tensor(x.iloc[:, 1:].values).float()
    y_id_col_name = y.axes[1][0]
    y = y.query(f"{y_id_col_name} in @common_keys")
    # noinspection PyCallingNonCallable
    y = torch.tensor(y.iloc[:, 1:].values).float()
    training_samples = utils_data.TensorDataset(x, y)
    data_loader = utils_data.DataLoader(training_samples, batch_size=len(x), shuffle=True)
    return x, data_loader, y


def get_parser():
    parser = argparse.ArgumentParser(
        description='TODO: ',
    )
    parser.add_argument('--x', action="store", default="x.csv")
    parser.add_argument('--y', action="store", default="y.csv")
    parser.add_argument('--gpu', action="store", type=bool, default=True)
    parser.add_argument('--epochs', action="store", type=int, default=100)
    parser.add_argument('--print_every', action="store", type=int, default=64)
    # parser.add_argument('--batch_size', action="store", type=int, default=64)
    return parser


main()
