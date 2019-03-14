import numpy
from typing import Any

from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import Dense

import networks
from common.options import TrainOptions, NetworkOptions
from networks.network import Network


class KerasNetwork(Network):

    model: Sequential
    data: dict
    train_options: TrainOptions

    def __init__(self):
        self.use_deterministic_behavior = False

    def init(self, data: dict, network_options: NetworkOptions, train_options: TrainOptions) -> None:
        self.train_options = train_options
        self.use_deterministic_behavior = self.train_options.deterministic \
            if train_options.deterministic is not None else False
        self._set_seed(train_options.seed)
        self.model = Sequential()
        hidden_layers = network_options.hidden_layer_sizes
        if hidden_layers is not None and len(hidden_layers) > 0:
            self.model.add(Dense(hidden_layers[0], input_dim=network_options.input_layer_size,
                                 activation=train_options.activation_function))
            for layer in hidden_layers[1:]:
                self.model.add(Dense(layer, activation=train_options.activation_function))
        else:
            self.model.add(Dense(network_options.output_layer_size, input_dim=network_options.input_layer_size,
                                 activation=train_options.activation_function))
        self.model.compile(loss=train_options.loss_function, optimizer=train_options.optimizer, metrics=['accuracy'])
        self.data = data

    def train(self) -> TrainOptions:
        batch_size = self.train_options.batch_size
        x = numpy.array(self.data['train'][0])
        y = numpy.array(self.data['train'][1])
        self.model.fit(x, y, batch_size if batch_size is not None else len(x), self.train_options.num_epochs)
        return self.train_options

    def validate(self) -> float:
        result = self.model.evaluate(numpy.array(self.data['valid'][0]), numpy.array(self.data['valid'][1]))
        return result[0]

    def predict(self, data) -> Any:
        return self.model.predict(numpy.array(data))[0][0]

    def save(self, path: str) -> None:
        self.model.save(str)

    def load(self, path: str) -> networks.network:
        self.model = load_model(str)
        return self

    def _set_seed(self, seed):
        seed = 0 if self.use_deterministic_behavior and seed is None else seed
        if seed is not None:
            numpy.random.seed(0)

