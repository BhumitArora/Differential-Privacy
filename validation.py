from numbers import Real, Integral

import numpy as np

from utils import warn_unused_args


def check_epsilon_delta(epsilon, delta, allow_zero=False):
    "epsilon delta"


def check_bounds(bounds, shape=0, min_separation=0.0, dtype=float):

    lower, upper = bounds

    if np.asarray(lower).size == 1 or np.asarray(upper).size == 1:
        lower = np.ravel(lower).astype(dtype)
        upper = np.ravel(upper).astype(dtype)
    else:
        lower = np.asarray(lower, dtype=dtype)
        upper = np.asarray(upper, dtype=dtype)

    n_bounds = lower.shape[0]

    for i in range(n_bounds):
        _lower = lower[i]
        _upper = upper[i]

        if _upper - _lower < min_separation:
            mid = (_upper + _lower) / 2
            lower[i] = mid - min_separation / 2
            upper[i] = mid + min_separation / 2

    if shape == 0:
        return lower.item(), upper.item()

    if n_bounds == 1:
        lower = np.ones(shape, dtype=dtype) * lower.item()
        upper = np.ones(shape, dtype=dtype) * upper.item()

    return lower, upper


def clip_to_norm(array, clip):

    norms = np.linalg.norm(array, axis=1) / clip
    norms[norms < 1] = 1

    return array / norms[:, np.newaxis]


def clip_to_bounds(array, bounds):

    lower, upper = check_bounds(bounds, np.size(bounds[0]), min_separation=0)
    clipped_array = array.copy()

    if np.allclose(lower, np.min(lower)) and np.allclose(upper, np.max(upper)):
        clipped_array = np.clip(clipped_array, np.min(lower), np.max(upper))
    else:

        for feature in range(array.shape[1]):
            clipped_array[:, feature] = np.clip(array[:, feature], lower[feature], upper[feature])

    return clipped_array

class DiffprivlibMixin:  
    _check_bounds = staticmethod(check_bounds)
    _clip_to_norm = staticmethod(clip_to_norm)
    _clip_to_bounds = staticmethod(clip_to_bounds)
    _warn_unused_args = staticmethod(warn_unused_args)

    def _validate_params(self):
        pass

    @staticmethod
    def _copy_parameter_constraints(cls, *args):
        if not hasattr(cls, "_parameter_constraints"):
            return {}

        return {k: cls._parameter_constraints[k] for k in args if k in cls._parameter_constraints}