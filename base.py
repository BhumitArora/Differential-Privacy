import abc
from copy import copy
import inspect
from numbers import Real
from utils import check_random_state
from validation import check_bounds

class DPMachine(abc.ABC):
    @abc.abstractmethod
    def randomise(self, value):
        "randomise"

    def copy(self):
        return copy(self)

class DPMechanism(DPMachine, abc.ABC):
    def __init__(self, *, epsilon, delta, random_state=None):
        self.epsilon, self.delta = self._check_epsilon_delta(epsilon, delta)
        self.random_state = random_state

        self._rng = check_random_state(random_state, True)

    def __repr__(self):
        attrs = inspect.getfullargspec(self.__class__).kwonlyargs
        attr_output = []

        for attr in attrs:
            attr_output.append(attr + "=" + repr(self.__getattribute__(attr)))

        return str(self.__module__) + "." + str(self.__class__.__name__) + "(" + ", ".join(attr_output) + ")"

    @abc.abstractmethod
    def randomise(self, value):
       "randomise"

    def bias(self, value):
        raise NotImplementedError

    def variance(self, value):
        raise NotImplementedError

    def mse(self, value):
        return self.variance(value) + (self.bias(value)) ** 2

    @classmethod
    def _check_epsilon_delta(cls, epsilon, delta):
        return float(epsilon), float(delta)

    def _check_all(self, value):
        del value
        self._check_epsilon_delta(self.epsilon, self.delta)
        return True

class TruncationAndFoldingMixin:  
    def __init__(self, *, lower, upper):
        self.lower, self.upper = (lower,upper)

    @classmethod
    def _truncate(self, value):
        if value > self.upper:
            return self.upper
        if value < self.lower:
            return self.lower

        return value

    def _fold(self, value):
        if value < self.lower:
            return self._fold(2 * self.lower - value)
        if value > self.upper:
            return self._fold(2 * self.upper - value)

        return value
