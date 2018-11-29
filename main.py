import argparse
import os
from math import fabs

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
    data_loader = get_dataloader(x, y)

    num_features_in = x.shape[1] - 1
    num_features_out = y.shape[1] - 1
    nw = Network(num_features_in, num_features_out, [4, 8], 0)  # TODO: get layers from input

    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(nw.parameters(), lr=0.001)
    optimizer = optim.SGD(nw.parameters(), lr=0.001, momentum=0.4)

    device = get_device(args)
    nw = nw.to(device)

    train(args, criterion, data_loader, device, nw, optimizer)
    # TODO: save

    validate(args, base_dir, device, nw, x, y)


def validate(args, base_dir, device, nw, x, y):
    x_valid = pd.read_csv(os.path.join(base_dir, args.xvalid))
    y_valid = pd.read_csv(os.path.join(base_dir, args.yvalid))
    data_loader = get_dataloader(x_valid, y_valid, 1)
    print("Validating...")
    loss = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        actual = predict(device, nw, data)
        current_loss = fabs(target.item() - actual.item())
        loss += current_loss
        print(f"input: {data.item()}, expected: {target.item()}, actual: {actual.item()}, loss: {current_loss}")
    print(f"total loss: {loss}")


def get_device(args):
    return torch.device("cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")


def predict(device, nw, data):
    nw.eval()
    with torch.no_grad():
        # noinspection PyCallingNonCallable
        output = nw.forward(torch.tensor(data).to(device).float())
    return output


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


def get_dataloader(x, y, batch_size = None):
    # noinspection PyUnusedLocal
    common_keys = pd.Index(x.iloc[:, 0]).intersection(pd.Index(y.iloc[:, 0]))
    x_id_col_name = x.axes[1][0]
    x = x.query(f"{x_id_col_name} in @common_keys").iloc[:, 1:].values
    # noinspection PyCallingNonCallable
    x = torch.FloatTensor(x)
    y_id_col_name = y.axes[1][0]
    y = y.query(f"{y_id_col_name} in @common_keys").iloc[:, 1:].values
    # noinspection PyCallingNonCallable
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
    parser.add_argument('--gpu', action="store", type=bool, default=True)
    parser.add_argument('--epochs', action="store", type=int, default=300)
    parser.add_argument('--print_every', action="store", type=int, default=64)
    # parser.add_argument('--batch_size', action="store", type=int, default=64)
    return parser


main()
