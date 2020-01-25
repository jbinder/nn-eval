from tensorflow.keras.callbacks import Callback


class EpochCountCallback(Callback):

    current_epoch: int

    def on_epoch_end(self, epoch, logs={}):
        self.current_epoch = epoch + 1

    def get_epic_count(self) -> int:
        return self.current_epoch
