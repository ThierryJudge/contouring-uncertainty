from typing import Literal


class ContourToMask:
    @staticmethod
    def __call__(landmarks, shape=(256, 256), labels=None, apply_argmax: bool = True,
                 reconstruction_type: Literal['spline', 'linear'] = 'spline'):
        raise NotImplementedError


class UMap:
    @staticmethod
    def __call__(mu, cov, labels=None):
        raise NotImplementedError


class SkewUMap:
    @staticmethod
    def __call__(mu, cov, alpha, labels=None):
        raise NotImplementedError