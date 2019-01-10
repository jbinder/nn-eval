import torch
from torch import nn
# noinspection PyPep8Naming
import torch.nn.functional as F


# inspired by https://github.com/udacity/DL_PyTorch
class FullyConnectedModel(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5, activation_function="relu", bias=True):
        super().__init__()
        self.activation_function = getattr(F, activation_function) if activation_function is not None else None
        if hidden_layers is not None and len(hidden_layers) > 0:
            self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0], bias)])
            layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
            self.hidden_layers.extend([nn.Linear(h1, h2, bias) for h1, h2 in layer_sizes])
            self.output = nn.Linear(hidden_layers[-1], output_size, bias)
        else:
            self.hidden_layers = []
            self.output = nn.Linear(input_size, output_size, bias)
        self.dropout = nn.Dropout(p=drop_p) if drop_p is not None else None

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation_function(layer(x)) if self.activation_function is not None else layer(x)
            x = self.dropout(x) if self.dropout is not None else x
        x = self.output(x)
        return x
