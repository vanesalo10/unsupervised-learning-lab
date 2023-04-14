import numpy as np


class PCA:
    def __init__(self, n_components=1) -> None:
        self.n_components = n_components
        self.eigenvector_subset = None

    def fit(self, input_matrix):
        self._input_matrix = input_matrix
        self.mean_matrix_value = np.mean(self._input_matrix, axis=0)
        centred_matrix = self._input_matrix - self.mean_matrix_value
        cov_mat = np.cov(centred_matrix , rowvar = False)
        eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)

        #sort the eigenvalues in descending order
        sorted_index = np.argsort(eigen_values)[::-1]
         
        sorted_eigenvectors = eigen_vectors[:,sorted_index]
        self.eigenvector_subset = sorted_eigenvectors[:, 0:self.n_components]
        return self

    def transform(self, input_matrix=None):
        self._input_matrix = input_matrix
        centred_matrix = self._input_matrix - self.mean_matrix_value
        return np.dot(np.transpose(self.eigenvector_subset),np.transpose(centred_matrix)).transpose()

    def fit_transform(self, input_matrix):
        fit = self.fit(input_matrix)
        return fit.transform(input_matrix)
