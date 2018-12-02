import os

import pandas as pd


class CsvDataProvider:
    def get_data_from_file(self, file_name_x, file_name_y):
        base_dir = os.getcwd()
        x = pd.read_csv(os.path.join(base_dir, file_name_x))
        y = pd.read_csv(os.path.join(base_dir, file_name_y))
        return self.get_data(x, y)

    def get_data(self, x, y):
        # noinspection PyUnusedLocal
        common_keys = pd.Index(x.iloc[:, 0]).intersection(pd.Index(y.iloc[:, 0]))
        x_id_col_name = x.axes[1][0]
        x = x.query(f"{x_id_col_name} in @common_keys").iloc[:, 1:].values
        y_id_col_name = y.axes[1][0]
        y = y.query(f"{y_id_col_name} in @common_keys").iloc[:, 1:].values
        return x, y
