import array

import matplotlib.pyplot as plt


class Visualizer:
    # noinspection PyMethodMayBeStatic
    def plot(self, x: array, y: array, predicted: array):
        plt.plot(x, y, 'go', label='expected', alpha=.5)
        plt.plot(x, predicted, label='prediction', alpha=0.5)
        plt.legend()
        plt.show()
