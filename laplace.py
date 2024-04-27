from base import TruncationAndFoldingMixin, DPMechanism
from utils import copy_docstring
import numpy as np


class Laplace(DPMechanism):
    def __init__(self, *, epsilon, delta=0.0, sensitivity, random_state=None):
        super().__init__(epsilon=epsilon, delta=delta, random_state=random_state)
        self.sensitivity = self._check_sensitivity(sensitivity)
        self._scale = None

    @classmethod
    def _check_sensitivity(cls, sensitivity):
        return float(sensitivity)

    def _check_all(self, value):
        super()._check_all(value)
        self._check_sensitivity(self.sensitivity)

        return True

    def bias(self, value):
        return 0.0

    def variance(self, value):
        self._check_all(0)

        return 2 * (self.sensitivity / (self.epsilon - np.log(1 - self.delta))) ** 2

    @staticmethod
    def _laplace_sampler(unif1, unif2, unif3, unif4):
        return np.log(1 - unif1) * np.cos(np.pi * unif2) + np.log(1 - unif3) * np.cos(np.pi * unif4)

    def randomise(self, value):
        self._check_all(value)

        scale = self.sensitivity / (self.epsilon - np.log(1 - self.delta))
        standard_laplace = self._laplace_sampler(self._rng.random(), self._rng.random(), self._rng.random(),
                                                 self._rng.random())

        return value - scale * standard_laplace
    
class LaplaceTruncated(Laplace, TruncationAndFoldingMixin):
    def __init__(self, *, epsilon, delta=0.0, sensitivity, lower, upper, random_state=None):
        super().__init__(epsilon=epsilon, delta=delta, sensitivity=sensitivity, random_state=random_state)
        TruncationAndFoldingMixin.__init__(self, lower=lower, upper=upper)

    @copy_docstring(Laplace.bias)
    def bias(self, value):
        self._check_all(value)

        shape = self.sensitivity / self.epsilon

        return shape / 2 * (np.exp((self.lower - value) / shape) - np.exp((value - self.upper) / shape))

    @copy_docstring(Laplace.variance)
    def variance(self, value):
        self._check_all(value)

        shape = self.sensitivity / self.epsilon

        variance = value ** 2 + shape * (self.lower * np.exp((self.lower - value) / shape)
                                         - self.upper * np.exp((value - self.upper) / shape))
        variance += (shape ** 2) * (2 - np.exp((self.lower - value) / shape)
                                    - np.exp((value - self.upper) / shape))

        variance -= (self.bias(value) + value) ** 2

        return variance

    def _check_all(self, value):
        # Laplace._check_all(self, value)
        # TruncationAndFoldingMixin._check_all(self, value)

        return True

    @copy_docstring(Laplace.randomise)
    def randomise(self, value):
        self._check_all(value)

        noisy_value = super().randomise(value)
        return self._truncate(noisy_value)
    