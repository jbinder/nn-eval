from normalizer.normalizer_base import NormalizerBase


class IdentityNormalizer(NormalizerBase):
    """ Normalizes values in a numpy array using the identity function. """

    @staticmethod
    def normalize(data):
        return NormalizerBase.normalize(data)

    @staticmethod
    def denormalize(data):
        return NormalizerBase.denormalize(data)
