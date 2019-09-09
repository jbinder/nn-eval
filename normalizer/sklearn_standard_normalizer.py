from sklearn.preprocessing import StandardScaler

from normalizer.normalizer_base import NormalizerBase


class SklearnStandardNormalizer(NormalizerBase):
    """ Normalizes values in a numpy array using scikit-learn. """

    def __init__(self, data):
        NormalizerBase._assert_is_numpy_array(data)
        data_prepared = self._prepare_data(data, data.ndim)
        self._scaler = SklearnStandardNormalizer._create_scaler(data_prepared)

    def normalize(self, data):
        if len(data) <= 0:
            return data
        ndim = data.ndim
        return self._unprepare_data(self._scaler.transform(self._prepare_data(data, ndim)), ndim)

    def denormalize(self, data):
        if len(data) <= 0:
            return data
        ndim = data.ndim
        return self._unprepare_data(self._scaler.inverse_transform(self._prepare_data(data, ndim)), ndim)

    @staticmethod
    def _create_scaler(data):
        scaler = StandardScaler()
        scaler.fit(data)
        return scaler

    @staticmethod
    def _prepare_data(data, ndim):
        if ndim == 1:
            return data.reshape(-1, 1)
        return data

    @staticmethod
    def _unprepare_data(data, ndim):
        if ndim == 1:
            return data.reshape(1, -1)[0]
        return data
