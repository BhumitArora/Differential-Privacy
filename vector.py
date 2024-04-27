from numbers import Real

import numpy as np

from base import DPMechanism
from utils import copy_docstring


class Vector(DPMechanism):
    def __init__(self, *, epsilon, function_sensitivity, data_sensitivity=1.0, dimension, alpha=0.01,
                 random_state=None):
        super().__init__(epsilon=epsilon, delta=0.0, random_state=random_state)
        self.function_sensitivity, self.data_sensitivity = self._check_sensitivity(function_sensitivity,
                                                                                   data_sensitivity)
        self.dimension = self._check_dimension(dimension)
        self.alpha = self._check_alpha(alpha)

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        if not delta == 0:
            raise ValueError("Delta must be zero")

        return super()._check_epsilon_delta(epsilon, delta)

    @classmethod
    def _check_alpha(cls, alpha):
        if not isinstance(alpha, Real):
            raise TypeError("Alpha must be numeric")

        if alpha <= 0:
            raise ValueError("Alpha must be strictly positive")

        return alpha

    @classmethod
    def _check_dimension(cls, vector_dim):
        if not isinstance(vector_dim, Real) or not np.isclose(vector_dim, int(vector_dim)):
            raise TypeError("d must be integer-valued")
        if int(vector_dim) < 1:
            raise ValueError("d must be strictly positive")

        return int(vector_dim)

    # @classmethod
    def _check_sensitivity(cls, function_sensitivity, data_sensitivity):
        if not isinstance(function_sensitivity, Real) or not isinstance(data_sensitivity, Real):
            raise TypeError("Sensitivities must be numeric")

        if function_sensitivity < 0 or data_sensitivity < 0:
            raise ValueError("Sensitivities must be non-negative")

        return function_sensitivity, data_sensitivity

    # def _check_all(self, value):
    #     super()._check_all(value)
    #     self._check_alpha(self.alpha)
    #     self._check_sensitivity(self.function_sensitivity, self.data_sensitivity)
    #     self._check_dimension(self.dimension)

    #     if not callable(value):
    #         raise TypeError("Value to be randomised must be a function")

    #     return True

    @copy_docstring(DPMechanism.bias)
    def bias(self, value):
        raise NotImplementedError

    @copy_docstring(DPMechanism.variance)
    def variance(self, value):
        raise NotImplementedError

    def randomise(self, value):
        self._check_all(value)

        epsilon_p = self.epsilon - 2 * np.log(1 + self.function_sensitivity * self.data_sensitivity /
                                              (0.5 * self.alpha))
        delta = 0

        if epsilon_p <= 0:
            delta = (self.function_sensitivity * self.data_sensitivity / (np.exp(self.epsilon / 4) - 1)
                     - 0.5 * self.alpha)
            epsilon_p = self.epsilon / 2

        scale = self.data_sensitivity * 2 / epsilon_p

        try:
            normed_noisy_vector = self._rng.standard_normal((self.dimension, 4)).sum(axis=1) / 2
            noisy_norm = self._rng.gamma(self.dimension / 4, scale, 4).sum()
        except AttributeError:  
            normed_noisy_vector = np.reshape([self._rng.normalvariate(0, 1) for _ in range(self.dimension * 4)],
                                             (-1, 4)).sum(axis=1) / 2
            noisy_norm = sum(self._rng.gammavariate(self.dimension / 4, scale) for _ in range(4)) if scale > 0 else 0.0

        norm = np.linalg.norm(normed_noisy_vector, 2)
        normed_noisy_vector = normed_noisy_vector / norm * noisy_norm

        def output_func(*args):
            input_vec = args[0]

            func = value(*args)

            if isinstance(func, tuple):
                func, grad = func
            else:
                grad = None

            func += np.dot(normed_noisy_vector, input_vec)
            func += 0.5 * delta * np.dot(input_vec, input_vec)

            if grad is not None:
                grad += normed_noisy_vector + delta * input_vec

                return func, grad

            return func

        return output_func