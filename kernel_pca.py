import numpy as np
from functools import reduce
from scipy.sparse.linalg import eigsh
from normalization import normalize

from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip
from sklearn.utils import validation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KernelCenterer
from pairwise import pairwise_kernels


class KernelPCA(TransformerMixin, BaseEstimator):
    def __init__(self, n_components, kernel_params, kernel_combine, alpha, normalization_method,
                 divide_after_combination=False, n_jobs=None):
        self.kernel_params = kernel_params
        self.kernel_combine = kernel_combine
        self.n_components = n_components
        self.alpha = alpha
        self.normalization_method = normalization_method
        self.n_jobs = n_jobs
        self.divide_after_combination = divide_after_combination

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        kernels = []
        for kernel, params in self.kernel_params.items():
            kernels.append(pairwise_kernels(X, Y, metric=kernel, filter_params=True, n_jobs=self.n_jobs, **params))
        if len(kernels) == 1 or self.kernel_combine is None:
            return kernels[0]
        if self.kernel_combine == '+':
            if self.alpha:
                return reduce((lambda x, y: (self.alpha * normalize(x, self.normalization_method) +
                                            (1 - self.alpha) * normalize(y, self.normalization_method)) / 2), kernels) \
                    if self.divide_after_combination else reduce((lambda x, y:
                                                                  self.alpha * normalize(x, self.normalization_method) +
                                            (1 - self.alpha) * normalize(y, self.normalization_method)), kernels)
            return reduce((lambda x, y: (x + y) / 2), kernels) if self.divide_after_combination else \
                reduce((lambda x, y: x + y), kernels)
        return reduce((lambda x, y: (x * y) / 2), kernels) if self.divide_after_combination else \
            reduce((lambda x, y: x * y), kernels)

    def _fit_transform(self, K):
        K = self._centerer.fit_transform(K)
        if self.n_components is None:
            n_components = K.shape[0]
        else:
            n_components = min(K.shape[0], self.n_components)
        random_state = check_random_state(None)
        v0 = random_state.uniform(-1, 1, K.shape[0])
        self.lambdas_, self.alphas_ = eigsh(K, n_components, which="LA", tol=0, maxiter=None, v0=v0)
        self.lambdas_ = validation._check_psd_eigenvalues(self.lambdas_, enable_warnings=False)
        self.alphas_, _ = svd_flip(self.alphas_, np.empty_like(self.alphas_).T)
        indices = self.lambdas_.argsort()[::-1]
        self.lambdas_ = self.lambdas_[indices]
        self.alphas_ = self.alphas_[:, indices]
        return K

    def fit(self, X):
        X = validation.check_array(X, accept_sparse='csr', copy=False)
        self._centerer = KernelCenterer()
        K = self._get_kernel(X)
        self._fit_transform(K)
        self.X_fit_ = X
        return self

    def fit_transform(self, X):
        self.fit(X)
        X_transformed = self.alphas_ * np.sqrt(self.lambdas_)
        return X_transformed

    def transform(self, X):
        validation.check_is_fitted(self)
        K = self._centerer.transform(self._get_kernel(X, self.X_fit_))
        non_zeros = np.flatnonzero(self.lambdas_)
        scaled_alphas = np.zeros_like(self.alphas_)
        scaled_alphas[:, non_zeros] = (self.alphas_[:, non_zeros] / np.sqrt(self.lambdas_[non_zeros]))
        return np.dot(K, scaled_alphas)
