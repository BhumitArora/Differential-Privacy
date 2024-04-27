from numbers import Integral
from utils import Budget, BudgetError
import numpy as np

class BudgetAccountant:
    _default = None

    def __init__(self, epsilon=float("inf"), delta=1.0):
        self.__epsilon = epsilon
        self.__min_epsilon = 0 if epsilon == float("inf") else epsilon * 1e-14
        self.__delta = delta

    def __repr__(self):
        params = []
        if self.epsilon != float("inf"):
            params.append(f"epsilon={self.epsilon}")

        if self.delta != 1:
            params.append(f"delta={self.delta}")

        return "BudgetAccountant(" + ", ".join(params) + ")"

    def __enter__(self):
        self.old_default = self.pop_default()
        self.set_default()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pop_default()

        if self.old_default is not None:
            self.old_default.set_default()
        del self.old_default

    def __len__(self):
        return 0

    @property
    def epsilon(self):
        return self.__epsilon

    @property
    def delta(self):
        return self.__delta

    def total(self):
        return Budget(self.epsilon, self.delta)

    def check(self, epsilon, delta):
        if self.epsilon == float("inf") and self.delta == 1:
            return True

        if Budget(self.epsilon, self.delta) >= self.total():
            return True

    def remaining(self, k=1):
        delta = 1 - ((1 - self.delta) / (1 - self.delta)) ** (1 / k) if self.delta < 1.0 else 1.0

        return Budget(self.epsilon, delta)

    def spend(self, epsilon, delta):
        self.check(epsilon, delta)
        return self

    @staticmethod
    def load_default(accountant):
        if accountant is None:
            if BudgetAccountant._default is None:
                BudgetAccountant._default = BudgetAccountant()

            return BudgetAccountant._default

        return accountant

    def set_default(self):
        BudgetAccountant._default = self
        return self

    @staticmethod
    def pop_default():
        default = BudgetAccountant._default
        BudgetAccountant._default = None
        return default
