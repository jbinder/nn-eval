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
        self.num_epochs = 0

    def run(self, data, network_options, train_options):
        data_loaders = {
            'train': self._get_data_loader(data['train'][0], data['train'][1]),
            'valid': self._get_data_loader(data['valid'][0], data['valid'][1], 1),
        }

        device = self._get_device(train_options.use_gpu)
        if self.use_deterministic_behavior:
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        self.nw = Network(
            network_options.input_layer_size,
            network_options.output_layer_size,
            network_options.hidden_layer_sizes,
            0)
        criterion = nn.MSELoss()
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(nw.parameters(), lr=0.001)
        self.optimizer = optim.SGD(self.nw.parameters(), lr=0.001, momentum=0.4)
        nw = self.nw.to(device)
        self.num_epochs = train_options.num_epochs
        self.train(train_options, criterion, data_loaders['train'], device)
        loss = self.validate(data_loaders['valid'], device, nw)
#        seeds = {
#            'torch': torch.initial_seed(),
#            'torch.cuda': torch.cuda.initial_seed(),
#        }
        parameters = [(param.data.cpu().numpy(), param.grad.cpu().numpy()) for param in list(nw.parameters())]
        return loss, parameters

    def validate(self, data_loader, device, nw):
        logging.info("Validating...")
        loss = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            actual = self.predict(device, nw, data)
            current_loss = fabs(target.item() - actual.item())
            loss += current_loss
            logging.info(f"input: {data.item()}, expected: {target.item()}, actual: {actual.item()}, loss: {current_loss}")
        logging.info(f"total loss: {loss}")
        return loss

    def predict(self, device, nw, data):
        nw.eval()
        with torch.no_grad():
            # noinspection PyCallingNonCallable
            output = nw.forward(torch.tensor(data).to(device).float())
        return output

    def train(self, options, criterion, data_loader, device):
        self.nw.train()
        for epoch in range(options.num_epochs):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(device)
                target = target.to(device)
                self.optimizer.zero_grad()
                outputs = self.nw(data)
                loss = criterion(outputs, target)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if batch_idx % options.print_every == 0:
                    info = "Epoch: {}/{}.. ".format(epoch + 1, options.num_epochs) + \
                           "\nProgress~: {:.2f}.. ".format(
                                ((1 + batch_idx) * len(data)) / (len(data_loader) * len(data)) * 100) + \
                           "\nTraining Loss: {:.3f}.. ".format(running_loss / options.print_every)
                    logging.info(info)
                    running_loss = 0.0

    def save(self, path: str) -> None:
        checkpoint = {'input_size': self.nw.hidden_layers[0].in_features,
                      'output_size': self.nw.output.out_features,
                      'hidden_layer_sizes': [each.out_features for each in self.nw.hidden_layers],
                      'optimizer_state': self.optimizer.state_dict(),
                      'epochs': self.num_epochs,
                      'state_dict': self.nw.state_dict()}
        torch.save(checkpoint, path)

    # def load(self, path: str) -> components.network:
    #     checkpoint = torch.load(path)
    #     model = PytorchNetwork()
    #     model.classifier = Network(checkpoint['input_size'], checkpoint['output_size'],
    #                                checkpoint['hidden_layer_sizes'])
    #     # TODO: optimizer.load_state_dict(checkpoint['optimizer_state'])
    #     model.load_state_dict(checkpoint['state_dict'])
    #     return model

    def _get_device(self, gpu):
        return torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")

    def _get_data_loader(self, x, y, batch_size=None):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        training_samples = utils_data.TensorDataset(x, y)
        batch_size = batch_size if batch_size is not None else len(x)
        return utils_data.DataLoader(training_samples, batch_size=batch_size,
                                     shuffle=not self.use_deterministic_behavior)
