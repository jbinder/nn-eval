import datetime
import os
from typing import Any

import numpy
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

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
    default_max_epochs: int
    visualize: bool

    def __init__(self):
        self.use_deterministic_behavior = False
        self.min_delta = 0.0
        self.default_max_epochs = 100000

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
            optimizer = self._get_optimizer(train_options.optimizer, train_options.learning_rate)
            self.model.compile(loss=train_options.loss_function, optimizer=optimizer)
        self.data = data
        self.visualize = train_options.visualize

    def train(self) -> TrainOptions:
        x = numpy.array(self.data['train'][0])
        y = numpy.array(self.data['train'][1])
        batch_size = self.train_options.batch_size
        batch_size = batch_size if batch_size is not None else len(x)
        max_epochs = self.train_options.num_epochs if self.train_options.num_epochs is not None \
            else self.default_max_epochs
        patience = self.train_options.progress_detection_patience \
            if self.train_options.progress_detection_patience else max_epochs
        early_stopping = EarlyStopping(monitor='loss', min_delta=self.min_delta, patience=patience, verbose=0,
                                       mode='auto', baseline=None, restore_best_weights=True)
        epoch_counter = EpochCountCallback()
        validation_data = None
        callbacks = [early_stopping, epoch_counter]
        if self.visualize:
            log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=1))
            validation_data = self._get_validation_data()
        self.model.fit(x, y, batch_size, max_epochs, validation_data=validation_data, callbacks=callbacks)
        # noinspection PyProtectedMember
        train_options = self.train_options._replace(num_epochs=epoch_counter.get_epic_count())
        return train_options

    def validate(self) -> float:
        x, y = self._get_validation_data()
        result = self.model.evaluate(numpy.array(x), numpy.array(y))
        return result

    def predict(self, data) -> Any:
        return self.model.predict(numpy.array(data))

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> networks.network:
        self.model = load_model(path)
        return self

    def _get_validation_data(self):
        x = self.data['valid'][0] if len(self.data['valid'][0]) > 0 else self.data['train'][0]
        y = self.data['valid'][1] if len(self.data['valid'][1]) > 0 else self.data['train'][1]
        return x, y

    def _set_seed(self, seed):
        seed = 0 if self.use_deterministic_behavior and seed is None else seed
        if seed is not None:
            numpy.random.seed(0)

    @staticmethod
    def _get_optimizer(optimizer_name: str, learning_rate: float):
        optimizer = getattr(tensorflow.keras.optimizers, optimizer_name)(lr=learning_rate)
        if optimizer_name == "sgd":
            optimizer.nesterov = True
            optimizer.momentum = 1
        return optimizer
