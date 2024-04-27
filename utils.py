
import secrets
import warnings

import numpy as np
from sklearn.utils import check_random_state as skl_check_random_state
from scipy.optimize import minimize_scalar

def utility_aware_sampling(dataset, alpha):
    original_dataset = dataset
    privacy_budget = 1.0 * len(original_dataset)
    iterations = []
    
    while len(dataset) > alpha * len(original_dataset):

        optimal_threshold = compute_optimal_threshold(dataset, privacy_budget)
        

        sampled_dataset = sample_dataset(dataset, optimal_threshold)
        

        estimation = compute_estimation(sampled_dataset)
        

        remaining_budget = privacy_budget - compute_budget_cost(sampled_dataset)
        

        iterations.append((optimal_threshold, estimation))
        
        if remaining_budget <= 0:
            break
        
        dataset = update_dataset(dataset, sampled_dataset)
        privacy_budget = remaining_budget
    

    final_output = aggregate_estimations(iterations)
    
    return final_output

def BW_function(t, Sr, epsilons, pis, omega_s, omega_n):
    summation_s = np.sum(epsilons * (1 - pis))
    summation_n = np.sum(epsilons - t)
    BW = omega_s * summation_s + omega_n * summation_n
    return BW

def compute_pi(epsilon, t):
    if epsilon < t:
        return (np.exp(epsilon) - 1) / (np.exp(t) - 1)
    else:
        return 1

def compute_optimal_threshold(Sr, epsilons, omega_s, omega_n):
    min_Sr = np.min(Sr)
    max_Sr = np.max(Sr)
    
    def objective_function(t):
        pis = np.array([compute_pi(epsilon, t) for epsilon in epsilons])
        return BW_function(t, Sr, epsilons, pis, omega_s, omega_n)
    
    result = minimize_scalar(objective_function, bounds=(min_Sr, max_Sr), method='bounded')
    return result.x

def sample_dataset(dataset, threshold):

    sampled_dataset = [tuple for tuple in dataset if np.random.uniform(0, 1) < threshold]
    return sampled_dataset

def compute_estimation(dataset):
    values = [x[0] for x in dataset]
    
    return sum(values) / len(values)

def compute_budget_cost(dataset):
    return 0.1

def update_dataset(original_dataset, sampled_dataset):
    updated_dataset = [tuple for tuple in original_dataset if tuple not in sampled_dataset]
    return updated_dataset

def aggregate_estimations(iterations):
    aggregated_estimation = sum(estimation for _, estimation in iterations)
    return aggregated_estimation

def warn_unused_args(args):
    if isinstance(args, str):
        args = [args]

    for arg in args:
        warnings.warn(f"Parameter '{arg}' is not functional in diffprivlib.  Remove this parameter to suppress this "
                      "warning.")
        
        
def copy_docstring(source):
    def copy_func(target):
        if source.__doc__ and not target.__doc__:
            target.__doc__ = source.__doc__
        return target
    return copy_func


def check_random_state(seed, secure=False):
    if secure:
        if isinstance(seed, secrets.SystemRandom):
            return seed

        if seed is None or seed is np.random.mtrand._rand:  
            return secrets.SystemRandom()
    elif isinstance(seed, secrets.SystemRandom):
        raise ValueError("secrets.SystemRandom instance cannot be passed when secure is False.")

    return skl_check_random_state(seed)


class Budget(tuple):
    def __new__(cls, epsilon, delta):
        if epsilon < 0:
            raise ValueError("Epsilon must be non-negative")

        if not 0 <= delta <= 1:
            raise ValueError("Delta must be in [0, 1]")

        return tuple.__new__(cls, (epsilon, delta))

    def __gt__(self, other):
        if self.__ge__(other) and not self.__eq__(other):
            return True
        return False

    def __ge__(self, other):
        if self[0] >= other[0] and self[1] >= other[1]:
            return True
        return False

    def __lt__(self, other):
        if self.__le__(other) and not self.__eq__(other):
            return True
        return False

    def __le__(self, other):
        if self[0] <= other[0] and self[1] <= other[1]:
            return True
        return False

    def __repr__(self):
        return f"(epsilon={self[0]}, delta={self[1]})"


class BudgetError(ValueError):
    'budgeterror'

