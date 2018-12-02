from math import fabs

import torch
import torch.utils.data as utils_data
from torch import nn, optim

from components.pytorch.fc_model import Network


class PyTorchNetwork:
    def __init__(self):
        self.use_deterministic_behavior = False

    def run(self, data, network_options, train_options):
        data_loaders = {
            'train': self._get_data_loader(data['train'][0], data['train'][1]),
            'valid': self._get_data_loader(data['valid'][0], data['valid'][1], 1),
        }

        device = self._get_device(train_options.use_gpu)
        if self.use_deterministic_behavior:
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
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
        self.train(train_options, criterion, data_loaders['train'], device, nw, optimizer)
        loss = self.validate(data_loaders['valid'], device, nw)
        return loss

    def validate(self, data_loader, device, nw):
        print("Validating...")
        loss = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            actual = self.predict(device, nw, data)
            current_loss = fabs(target.item() - actual.item())
            loss += current_loss
            print(f"input: {data.item()}, expected: {target.item()}, actual: {actual.item()}, loss: {current_loss}")
        print(f"total loss: {loss}")
        return loss

    def predict(self, device, nw, data):
        nw.eval()
        with torch.no_grad():
            # noinspection PyCallingNonCallable
            output = nw.forward(torch.tensor(data).to(device).float())
        return output

    def train(self, options, criterion, data_loader, device, nw, optimizer):
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

    def _get_device(self, gpu):
        return torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")

    def _get_data_loader(self, x, y, batch_size=None):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        training_samples = utils_data.TensorDataset(x, y)
        batch_size = batch_size if batch_size is not None else len(x)
        return utils_data.DataLoader(training_samples, batch_size=batch_size,
                                     shuffle=not self.use_deterministic_behavior)

