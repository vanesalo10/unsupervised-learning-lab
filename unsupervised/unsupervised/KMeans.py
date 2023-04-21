import numpy as np


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=0.0001, init_method='random', random_state=None, n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method
        self.random_state = random_state
        self.n_init = n_init

    def _initialize_centroids(self, X):
        if self.init_method == 'random':
            np.random.seed(self.random_state)
            centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.init_method == 'kmeans++':
            np.random.seed(self.random_state)
            centroids = np.zeros((self.n_clusters, X.shape[1]))
            centroids[0] = X[np.random.choice(X.shape[0], 1)]
            for i in range(1, self.n_clusters):
                distances = np.zeros((X.shape[0], i))
                for j in range(i):
                    distances[:, j] = np.linalg.norm(X - centroids[j], axis=1)
                min_distances = np.min(distances, axis=1)
                min_distances_squared = min_distances ** 2
                probabilities = min_distances_squared / np.sum(min_distances_squared)
                centroids[i] = X[np.random.choice(X.shape[0], 1, p=probabilities)]
        return centroids

    def _compute_distances(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for j in range(self.n_clusters):
            distances[:, j] = np.linalg.norm(X - self.centroids[j], axis=1)
        return distances

    def _assign_clusters(self, X):
        distances = np.sum((X[:, np.newaxis, :] - self.centroids) ** 2, axis=2)
        return np.argmin(distances, axis=1)


    def _update_centroids(self, X, cluster_labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for j in range(self.n_clusters):
            centroids[j] = np.mean(X[cluster_labels == j], axis=0)
        return centroids

    def _compute_inertia(self, X, cluster_labels):
        distances = self._compute_distances(X)
        return np.sum(np.min(distances, axis=1))

    def fit(self, X):
        best_inertia = np.inf
        for n in range(self.n_init):
            self.centroids = self._initialize_centroids(X)
            for i in range(self.max_iter):
                cluster_labels = self._assign_clusters(X)
                prev_centroids = self.centroids.copy()
                self.centroids = self._update_centroids(X, cluster_labels)
                if np.allclose(prev_centroids, self.centroids, rtol=0, atol=self.tol):
                    break
            inertia = self._compute_inertia(X, cluster_labels)
            if inertia < best_inertia:
                best_inertia = inertia
                self.best_centroids = self.centroids.copy()

    def fit_transform(self, X):
        self.fit(X)
        distances = self._compute_distances(X)
        return distances

    def transform(self, X):
        distances = self._compute_distances(X)
        return distances