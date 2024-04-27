import numpy as np
from joblib import delayed, Parallel
from scipy import optimize
from sklearn import linear_model

from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn._loss import HalfBinomialLoss
SKL_LOSS_MODULE = True

from Budgetaccountant import BudgetAccountant
from vector import Vector
from validation import DiffprivlibMixin

class LogisticRegression(linear_model.LogisticRegression, DiffprivlibMixin):
    _parameter_constraints = DiffprivlibMixin._copy_parameter_constraints(
        linear_model.LogisticRegression, "tol", "C", "fit_intercept", "max_iter", "verbose", "warm_start", "n_jobs",
        "random_state")

    def __init__(self, *, epsilon=1.0, data_norm=None, tol=1e-4, C=1.0, fit_intercept=True, max_iter=100, 
                 random_state=None, accountant=BudgetAccountant(epsilon=0.01, delta=0), **unused_args):
        
        super().__init__(penalty='l2', dual=False, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=1.0,
                         class_weight=None, random_state=random_state, solver='lbfgs', max_iter=max_iter,
                         multi_class='ovr')
        
        self.epsilon = epsilon
        self.data_norm = data_norm
        self.classes_ = None
        self.accountant = BudgetAccountant.load_default(accountant)

    def fit(self, X, y, sample_weight=None):
        self._validate_params()
        self.accountant.check(self.epsilon, 0)

        if sample_weight is not None:
            self._warn_unused_args("sample_weight")

        X, y = self._validate_data(X, y, accept_sparse='csr', dtype=float, order="C",
                                   accept_large_sparse=True)

        self.classes_ = np.unique(y)
        _, n_features = X.shape

        if self.data_norm is None:
            self.data_norm = np.linalg.norm(X, axis=1).max()

        X = self._clip_to_norm(X, self.data_norm)

        n_classes = len(self.classes_)
        classes_ = self.classes_

        if len(self.classes_) == 2:
            n_classes = 1
            classes_ = classes_[1:]

        if self.warm_start:
            warm_start_coef = getattr(self, 'coef_', None)
        else:
            warm_start_coef = None
        if warm_start_coef is not None and self.fit_intercept:
            warm_start_coef = np.append(warm_start_coef, self.intercept_[:, np.newaxis], axis=1)

        self.coef_ = []
        self.intercept_ = np.zeros(n_classes)

        if warm_start_coef is None:
            warm_start_coef = [None] * n_classes

        path_func = delayed(_logistic_regression_path)

        fold_coefs_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, prefer='processes')(
            path_func(X, y, epsilon=self.epsilon / n_classes, data_norm=self.data_norm, pos_class=class_, Cs=[self.C],
                      fit_intercept=self.fit_intercept, max_iter=self.max_iter, tol=self.tol, verbose=self.verbose,
                      coef=warm_start_coef_, check_input=False)
            for class_, warm_start_coef_ in zip(classes_, warm_start_coef))

        fold_coefs_, _, n_iter_ = zip(*fold_coefs_)
        self.n_iter_ = np.asarray(n_iter_, dtype=np.int32)[:, 0]

        self.coef_ = np.asarray(fold_coefs_)
        self.coef_ = self.coef_.reshape(n_classes, n_features + int(self.fit_intercept))

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]

        self.accountant.spend(self.epsilon, 0)

        return self


def _logistic_regression_path(X, y, epsilon, data_norm, pos_class=None, Cs=10, fit_intercept=True, max_iter=100,
                              tol=1e-4, verbose=0, coef=None, random_state=None, check_input=True, **unused_args):
    import numbers
    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, int(Cs))

    if fit_intercept:
        data_norm = np.sqrt(data_norm ** 2 + 1)
    n_samples, n_features = X.shape

    classes = np.unique(y)

    if pos_class is None:
        if classes.size > 2:
            raise ValueError('To fit OvR, use the pos_class argument')

        pos_class = classes[1]

    sample_weight = np.ones(X.shape[0], dtype=X.dtype)

    output_vec = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)
    mask = (y == pos_class)
    y_bin = np.ones(y.shape, dtype=X.dtype)
    y_bin[~mask] = 0.0 if SKL_LOSS_MODULE else -1.0

    if coef is not None:
        if coef.size not in (n_features, output_vec.size):
            raise ValueError(f"Initialization coef is of shape {coef.size}, expected shape {n_features} or "
                             f"{output_vec.size}")
        output_vec[:coef.size] = coef

    target = y_bin


    func = LinearModelLoss(base_loss=HalfBinomialLoss(), fit_intercept=fit_intercept).loss_gradient
    sw_sum = n_samples
 

    coefs = []
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    for i, C in enumerate(Cs):
        l2_reg_strength = 1.0 / (C * sw_sum)
        vector_mech = Vector(epsilon=epsilon, dimension=n_features + int(fit_intercept), alpha=l2_reg_strength,
                             function_sensitivity=0.25, data_sensitivity=data_norm, random_state=random_state)
        noisy_logistic_loss = vector_mech.randomise(func)

        args = (X, target, sample_weight, l2_reg_strength) if SKL_LOSS_MODULE else (X, target, l2_reg_strength,
                                                                                    sample_weight)

        iprint = [-1, 50, 1, 100, 101][np.searchsorted(np.array([0, 1, 2, 3]), verbose)]
        output_vec, _, info = optimize.fmin_l_bfgs_b(noisy_logistic_loss, output_vec, fprime=None,
                                                     args=args, iprint=iprint, pgtol=tol, maxiter=max_iter)
        coefs.append(output_vec.copy())

        n_iter[i] = info['nit']

    return np.array(coefs), np.array(Cs), n_iter
