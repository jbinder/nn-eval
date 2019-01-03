import os

import pandas as pd
import numpy as np


class CsvDataProvider:
    def get_data_from_file(self, file_name_x, file_name_y, train_percentage=0.7):
        base_dir = os.getcwd()
        x = pd.read_csv(os.path.join(base_dir, file_name_x))
        y = pd.read_csv(os.path.join(base_dir, file_name_y))
        return self._get_data(x, y, train_percentage)

    @staticmethod
    def _get_data(x, y, train_percentage):
        # noinspection PyUnusedLocal
        common_keys = pd.Index(x.iloc[:, 0]).intersection(pd.Index(y.iloc[:, 0]))
        x_id_col_name = x.axes[1][0]
        x = x.query(f"{x_id_col_name} in @common_keys").iloc[:, 1:].values
        y_id_col_name = y.axes[1][0]
        y = y.query(f"{y_id_col_name} in @common_keys").iloc[:, 1:].values
        x_train, x_valid = np.split(x, [int(train_percentage * len(x))])
        y_train, y_valid = np.split(y, [int(train_percentage * len(y))])
        return {'train': (x_train, y_train), 'valid': (x_valid, y_valid)}
