import logging
from math import fabs

import torch
import torch.utils.data as utils_data
from torch import nn, optim

import components
from components.network import Network as ANetwork
from components.pytorch.fc_model import Network


class PytorchNetwork(ANetwork):
    def __init__(self):
        super().__init__()
        self.use_deterministic_behavior = False
        self.nw = None
        self.optimizer = None
        self.criterion = None
        self.train_options = None
        self.data_loaders = None
        self.device = None

    def init(self, data, network_options, train_options):
        self.data_loaders = {
            'train': self._get_data_loader(data['train'][0], data['train'][1]),
            'valid': self._get_data_loader(data['valid'][0], data['valid'][1], 1),
        }

        self.device = self._get_device(train_options.use_gpu)
        if self.use_deterministic_behavior:
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        self.nw = Network(
            network_options.input_layer_size,
            network_options.output_layer_size,
            network_options.hidden_layer_sizes,
            0)
        self.criterion = nn.MSELoss()
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(nw.parameters(), lr=0.001)
        self.optimizer = optim.SGD(self.nw.parameters(), lr=0.001, momentum=0.4)
        self.nw = self.nw.to(self.device)
        self.train_options = train_options
#        seeds = {
#            'torch': torch.initial_seed(),
#            'torch.cuda': torch.cuda.initial_seed(),
#        }

    def validate(self) -> float:
        logging.info("Validating...")
        loss = 0
        for batch_idx, (data, target) in enumerate(self.data_loaders['valid']):
            actual = self.predict(data)
            current_loss = fabs(target.item() - actual)
            loss += current_loss
            logging.info(f"input: {data.item()}, expected: {target.item()}, actual: {actual},"
                         f"loss: {current_loss}")
        logging.info(f"total loss: {loss}")
        return loss

    def predict(self, data):
        if not self.device:  # TODO: fallback, remove
            self.device = self._get_device(True)
        self.nw.eval()
        with torch.no_grad():
            # noinspection PyCallingNonCallable
            output = self.nw.forward(torch.tensor(data).to(self.device).float())
        return output.item()

    def train(self):
        self.nw.train()
        data_loader = self.data_loaders['train']
        for epoch in range(self.train_options.num_epochs):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.nw(data)
                loss = self.criterion(outputs, target)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if batch_idx % self.train_options.print_every == 0:
                    info = "Epoch: {}/{}.. ".format(epoch + 1, self.train_options.num_epochs) + \
                           "\nProgress~: {:.2f}.. ".format(
                                ((1 + batch_idx) * len(data)) / (len(data_loader) * len(data)) * 100) + \
                           "\nTraining Loss: {:.3f}.. ".format(running_loss / self.train_options.print_every)
                    logging.info(info)
                    running_loss = 0.0

    def save(self, path: str) -> None:
        checkpoint = {'input_size': self.nw.hidden_layers[0].in_features,
                      'output_size': self.nw.output.out_features,
                      'hidden_layer_sizes': [each.out_features for each in self.nw.hidden_layers],
                      'optimizer_state': self.optimizer.state_dict(),
                      'epochs': self.train_options.num_epochs,
                      'state_dict': self.nw.state_dict()}
        torch.save(checkpoint, path)

    def load(self, path: str) -> components.network:
        checkpoint = torch.load(path)
        # TODO: make static and load to new network?
        #model = PytorchNetwork()
        #model.nw = Network(checkpoint['input_size'], checkpoint['output_size'],
        #                   checkpoint['hidden_layer_sizes'])
        # TODO: optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.nw.load_state_dict(checkpoint['state_dict'])
        return self

    def _get_device(self, gpu):
        return torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")

    def _get_data_loader(self, x, y, batch_size=None):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        training_samples = utils_data.TensorDataset(x, y)
        batch_size = batch_size if batch_size is not None else len(x)
        return utils_data.DataLoader(training_samples, batch_size=batch_size,
                                     shuffle=not self.use_deterministic_behavior)

