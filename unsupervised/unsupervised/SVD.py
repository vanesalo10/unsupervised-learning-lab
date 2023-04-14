from numpy.linalg import svd
import numpy as np


class SVD:
    def __init__(self, full_matrices=True, n_singular_values=1) -> None:
        self._input_matrix = None
        self.u = None
        self.sigma = None
        self.v_t = None
        self.n_singular_values = n_singular_values
        self.full_matrices = full_matrices

    def fit(self, input_matrix=None):

        self._input_matrix = input_matrix

        self.u, self.sigma, self.v_t = svd(
            self._input_matrix, full_matrices=self.full_matrices)

        self.sigma_modified = np.zeros((self.u.shape[1], self.v_t.shape[0]))

        for n in range(self.n_singular_values):
            self.sigma_modified[n, n] = self.sigma[n]

        self.mean_matrix_value = np.mean(input_matrix, axis=0)
        return self

    def transform(self, input_matrix=None):

        self._input_matrix = input_matrix

        return np.dot(self.u, np.dot(self.sigma_modified, self.v_t))

    def fit_transform(self, input_matrix=None):
        fit = self.fit(input_matrix)
        return fit.transform(input_matrix)


class TruncateSVD:
    def __init__(self, full_matrices=True, n_components=2) -> None:
        self._input_matrix = None
        self.u = None
        self.sigma = None
        self.v_t = None
        self.n_components = n_components
        self.full_matrices = full_matrices

    def fit(self, input_matrix=None):
        self._input_matrix = input_matrix
        self.u, self.sigma, self.v_t = np.linalg.svd(
            self._input_matrix, full_matrices=self.full_matrices)
        self.components_ = self.v_t[:self.n_components]
        return self

    def transform(self, input_matrix=None):
        self._input_matrix = input_matrix
        return np.dot(self._input_matrix, self.components_.T)

    def fit_transform(self, input_matrix=None):
        fit = self.fit(input_matrix)
        return fit.transform(input_matrix)
