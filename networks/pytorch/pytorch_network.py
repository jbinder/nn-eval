import array
import logging
from typing import Any

import torch
import torch.utils.data as utils_data
from torch import nn, optim
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

import networks
from common.options import TrainOptions
from networks.network import Network as ANetwork
from networks.pytorch.fully_connected_model import FullyConnectedModel


class PytorchNetwork(ANetwork):

    use_deterministic_behavior: bool
    nw: torch.nn.Module
    optimizer: Optimizer
    criterion: _Loss
    validation_criterion: _Loss
    train_options: TrainOptions
    data_loaders: array
    device: torch.device
    max_epochs: int
    default_max_epochs: int
    progress_detection_epoch_count: int

    def __init__(self):
        super().__init__()
        self.use_deterministic_behavior = False
        self.nw = None
        self.optimizer = None
        self.criterion = None
        self.validation_criterion = None
        self.train_options = None
        self.data_loaders = None
        self.device = None
        self.default_max_epochs = 100000
        self.progress_detection_epoch_count = 100
        self.progress_detection_min_delta = 0.0

    def init(self, data, network_options, train_options):
        self.device = self._get_device(train_options.use_gpu)
        logging.info(f"Using device: {self.device}")

        self.train_options = train_options
        self.use_deterministic_behavior = self.train_options.deterministic \
            if train_options.deterministic is not None else False

        self.data_loaders = {
            'train': self._get_data_loader(data['train'][0], data['train'][1], train_options.batch_size),
            'valid': self._get_data_loader(data['valid'][0], data['valid'][1], 1),
        }

        self._set_seed(train_options.seed)
        self.max_epochs = self.train_options.num_epochs if self.train_options.num_epochs is not None \
            else self.default_max_epochs
        self.progress_detection_epoch_count = train_options.progress_detection_patience \
            if train_options.progress_detection_patience is not None else self.max_epochs
        self.progress_detection_min_delta = train_options.progress_detection_min_delta \
            if train_options.progress_detection_min_delta is not None else self.progress_detection_min_delta
        self.nw = FullyConnectedModel(
            network_options.input_layer_size,
            network_options.output_layer_size,
            network_options.hidden_layer_sizes,
            train_options.dropout_rate, train_options.activation_function, train_options.bias)
        if train_options.loss_function is not None:
            self.criterion = self._get_loss_function(train_options.loss_function)
        self.validation_criterion = nn.MSELoss()
        if train_options.optimizer is not None:
            self.optimizer = self._get_optimizer(train_options.optimizer, train_options.learning_rate)
        self.nw = self.nw.to(self.device)

    def validate(self) -> float:
        logging.info("Validating...")
        loss = 0
        data_loader = self.data_loaders['valid'] if len(self.data_loaders['valid']) > 0 else self.data_loaders['train']
        for batch_idx, (data, target) in enumerate(data_loader):
            target = target.to(self.device)
            actual = self._forward(data)
            current_loss = self.validation_criterion(actual, target).item()
            loss += current_loss
            logging.info(f"input: {data.numpy()}, expected: {target.cpu().numpy()}, actual: {actual.cpu().numpy()},"
                         f"loss: {current_loss}")
        logging.info(f"total loss: {loss}")
        return loss

    def predict(self, data) -> Any:
        output = self._forward(data)
        return output.cpu().numpy()

    def _forward(self, data):
        self.nw.eval()
        with torch.no_grad():
            # noinspection PyCallingNonCallable
            output = self.nw.forward(torch.tensor(data).to(self.device).float())
        return output

    def train(self) -> TrainOptions:
        self.nw.train()
        data_loader = self.data_loaders['train']
        last_losses = []
        current_losses = []
        current_losses_num_epochs = 1
        for epoch in range(1, self.max_epochs):
            running_loss = 0.0
            current_epoch_loss = 0.0
            current_epoch_batch_idx = 0
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.nw(data)
                loss = self.criterion(outputs, target)
                loss.backward()
                # noinspection PyArgumentList
                self.optimizer.step()
                running_loss += loss.item()
                current_epoch_loss += loss.item()
                current_epoch_batch_idx += 1
                if batch_idx % self.train_options.print_every == 0:
                    info = "Epoch: {}/{}.. ".format(epoch, self.max_epochs) + \
                           "\nProgress~: {:.2f}.. ".format(
                                ((1 + batch_idx) * len(data)) / (len(data_loader) * len(data)) * 100) + \
                           "\nTraining Loss: {:.10f}.. ".format(running_loss / (batch_idx + 1))
                    logging.info(info)
                    running_loss = 0.0
                if current_losses_num_epochs > self.progress_detection_epoch_count:
                    current_min = min(current_losses)
                    last_min = min(last_losses) if len(last_losses) > 0 else None
                    last_delta = last_min - current_min if len(last_losses) > 0 else 0
                    logging.info(f"Progress detection: delta={last_delta}, current min={current_min}, "
                                 f"last min={last_min}")
                    if len(last_losses) < 1 or (last_delta > self.progress_detection_min_delta):
                        last_losses.clear()
                        last_losses.extend(current_losses)
                        current_losses.clear()
                        current_losses_num_epochs = 1
                    else:
                        logging.info("No more progress, done.")
                        return self._get_train_options(data_loader.batch_size, epoch)
            current_losses.append(current_epoch_loss / current_epoch_batch_idx)
            current_losses_num_epochs = current_losses_num_epochs + 1
        return self._get_train_options(data_loader.batch_size, self.max_epochs)

    def _set_seed(self, seed):
        seed = 0 if self.use_deterministic_behavior and seed is None else seed
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _get_train_options(self, batch_size, epochs):
        return self.train_options._replace(num_epochs=epochs, batch_size=batch_size,
                                           use_gpu=self._is_gpu(str(self.device)), seed=torch.initial_seed())

    def save(self, path: str) -> None:
        input_size = self.nw.hidden_layers[0].in_features if len(self.nw.hidden_layers) > 0 \
            else self.nw.output.in_features
        checkpoint = {'input_size': input_size,
                      'output_size': self.nw.output.out_features,
                      'hidden_layer_sizes': [each.out_features for each in self.nw.hidden_layers],
                      'optimizer_state': self.optimizer.state_dict(),
                      'train_options': self.train_options,
                      'state_dict': self.nw.state_dict(),
                      }
        torch.save(checkpoint, path)

    def load(self, path: str) -> networks.network:
        checkpoint = torch.load(path)
        train_options = checkpoint['train_options']
        self.nw = PytorchNetwork()
        self.nw = FullyConnectedModel(checkpoint['input_size'], checkpoint['output_size'],
                                      checkpoint['hidden_layer_sizes'], train_options.dropout_rate,
                                      train_options.activation_function, train_options.bias)
        if not self.device:  # TODO: fallback, remove
            self.device = self._get_device(True)
        self.nw = self.nw.to(self.device)
        self.optimizer = self._get_optimizer(train_options.optimizer, train_options.learning_rate)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.nw.load_state_dict(checkpoint['state_dict'])
        return self

    @staticmethod
    def _get_device(gpu):
        return torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")

    @staticmethod
    def _is_gpu(device):
        return True if "cuda" in device else False

    def _get_data_loader(self, x, y, batch_size=None):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        training_samples = utils_data.TensorDataset(x, y)
        batch_size = batch_size if batch_size is not None else len(x)
        return utils_data.DataLoader(training_samples, batch_size=batch_size,
                                     shuffle=not self.use_deterministic_behavior)

    def _get_optimizer(self, optimizer: str, learning_rate: float):
        return getattr(optim, optimizer)(self.nw.parameters(), lr=learning_rate)  # TODO: set momentum for SGD?

    @staticmethod
    def _get_loss_function(loss_function):
        return getattr(nn, loss_function)()
