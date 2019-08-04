from normalizer.normalizer_base import NormalizerBase


class IdentityNormalizer(NormalizerBase):
    """ Normalizes values in a numpy array using the identity function. """

    def normalize(self, data):
        return NormalizerBase.normalize(data)

    def denormalize(self, data):
        return NormalizerBase.denormalize(data)
