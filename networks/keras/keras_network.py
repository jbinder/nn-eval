from typing import Any

import keras
import numpy
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.engine.saving import load_model
from keras.layers import Dense

import networks
from common.options import TrainOptions, NetworkOptions
from networks.keras.callbacks.epoch_count_callback import EpochCountCallback
from networks.network import Network


class KerasNetwork(Network):
    model: Sequential
    data: dict
    train_options: TrainOptions
    use_deterministic_behavior: bool
    min_delta: float
    learning_rate: float

    def __init__(self):
        self.use_deterministic_behavior = False
        self.min_delta = 0.0
        self.learning_rate = 0.001

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
            self.model.add(Dense(network_options.output_layer_size, activation=train_options.activation_function))
        else:
            self.model.add(Dense(network_options.output_layer_size, input_dim=network_options.input_layer_size,
                                 activation=train_options.activation_function))
        if train_options.optimizer is not None:
            optimizer = self._get_optimizer(train_options.optimizer)
            self.model.compile(loss=train_options.loss_function, optimizer=optimizer, metrics=['accuracy'])
        self.data = data

    def train(self) -> TrainOptions:
        x = numpy.array(self.data['train'][0])
        y = numpy.array(self.data['train'][1])
        batch_size = self.train_options.batch_size
        batch_size = batch_size if batch_size is not None else len(x)
        early_stopping = EarlyStopping(monitor='loss', min_delta=self.min_delta, patience=2, verbose=0, mode='auto',
                                       baseline=None, restore_best_weights=False)
        epoch_counter = EpochCountCallback()
        self.model.fit(x, y, batch_size, self.train_options.num_epochs, callbacks=[early_stopping, epoch_counter])
        train_options = self.train_options._replace(num_epochs=epoch_counter.get_epic_count())
        return train_options

    def validate(self) -> float:
        result = self.model.evaluate(numpy.array(self.data['valid'][0]), numpy.array(self.data['valid'][1]))
        return result[0]

    def predict(self, data) -> Any:
        return self.model.predict(numpy.array(data))[0][0]

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> networks.network:
        self.model = load_model(path)
        return self

    def _set_seed(self, seed):
        seed = 0 if self.use_deterministic_behavior and seed is None else seed
        if seed is not None:
            numpy.random.seed(0)

    def _get_optimizer(self, optimizer: str):
        return getattr(keras.optimizers, optimizer)(lr=self.learning_rate)
