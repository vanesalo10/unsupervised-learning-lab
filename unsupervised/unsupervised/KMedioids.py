import numpy as np


def _compute_inertia(distances):
    inertia = np.sum(np.min(distances, axis=1))
    return inertia


class KMedoids():
    def __init__(self, n_clusters=8, metric="euclidean", method="alternate",init="random",max_iter=300,random_state=None,):
        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state

    def check_random_state(self, random_state):
        if random_state is None or isinstance(random_state, int):
            return np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            return random_state
        else:
            raise ValueError("Invalid random state")

    def _check_nonnegative_int(self, value, desc, strict=True):
        if strict:
            negative = (value is None) or (value <= 0)
        else:
            negative = (value is None) or (value < 0)
        if negative or not isinstance(value, (int, np.integer)):
            raise

    def fit(self, X, y=None):
        random_state_ = self.check_random_state(self.random_state)
        if X.dtype != np.float64:
            X = X.astype(np.float64)

        self.n_features_in_ = X.shape[1]
        if self.n_clusters > X.shape[0]:
            raise

        if self.metric == 'euclidean':
            D = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
        elif self.metric == 'manhattan':
            D = np.sum(np.abs(X[:, np.newaxis, :] - X[np.newaxis, :, :]), axis=2)
        else:
            raise ValueError(f"Invalid metric '{self.metric}'. Valid options are: 'euclidean', 'manhattan'")

        medoid_idxs = self._initialize_medoids(
            D, self.n_clusters, random_state_, X
        )
        labels = None

        for self.n_iter_ in range(0, self.max_iter):
            old_medoid_idxs = np.copy(medoid_idxs)
            labels = np.argmin(D[medoid_idxs, :], axis=0)

            if self.method == "alternate":
                self._update_medoid_idxs_in_place(D, labels, medoid_idxs)
            else:
                raise

        self.cluster_centers_ = X[medoid_idxs]
        self.labels_ = np.argmin(D[medoid_idxs, :], axis=0)
        self.medoid_indices_ = medoid_idxs
        self.inertia_ = _compute_inertia(self.transform(X))

        return self

    def _update_medoid_idxs_in_place(self, D, labels, medoid_idxs):
        for k in range(self.n_clusters):
            cluster_k_idxs = np.where(labels == k)[0]
            if len(cluster_k_idxs) == 0:
                continue
            in_cluster_distances = D[cluster_k_idxs, cluster_k_idxs[:, np.newaxis]]
            in_cluster_all_costs = np.sum(in_cluster_distances, axis=1)
            min_cost_idx = np.argmin(in_cluster_all_costs)
            min_cost = in_cluster_all_costs[min_cost_idx]
            curr_cost = in_cluster_all_costs[np.argmax(cluster_k_idxs == medoid_idxs[k])]
            if min_cost < curr_cost:
                medoid_idxs[k] = cluster_k_idxs[min_cost_idx]

    def _compute_cost(self, D, medoid_idxs):
        return _compute_inertia(D[:, medoid_idxs])

    def transform(self, X):
        if X.dtype != np.float64:
            X = X.astype(np.float64)
        DXY = self.pairwise_distances(X, Y=self.cluster_centers_, metric=self.metric)
        return DXY

    def pairwise_distances(self, X, Y, metric=None):
        if metric == 'euclidean':
            DXY = np.sqrt(((X[:, None] - Y) ** 2).sum(axis=2))
        elif metric == 'manhattan':
            DXY = np.abs(X[:, None] - Y).sum(axis=2)
        else:
            raise ValueError("Metric not supported")
        return DXY

    def pairwise_distances_argmin(self, X, Y, metric=None):
        if metric == 'euclidean':
            pd_argmin = np.argmin(((X[:, None] - Y) ** 2).sum(axis=2), axis=1)
        elif metric == 'manhattan':
            pd_argmin = np.argmin(np.abs(X[:, None] - Y).sum(axis=2), axis=1)
        else:
            raise ValueError("Metric not supported")
        return pd_argmin

    def predict(self, X):
        if X.dtype != np.float64:
            X = X.astype(np.float64)
        pd_argmin = self.pairwise_distances_argmin(X, Y=self.cluster_centers_, metric=self.metric)
        return pd_argmin

    def fit_transform(self, X):
        self.fit(X)
        if X.dtype != np.float64:
            X = X.astype(np.float64)
        pd_argmin = self.pairwise_distances_argmin(X, Y=self.cluster_centers_, metric=self.metric)
        return pd_argmin

    def _initialize_medoids(self, D, n_clusters, random_state_, X=None):
        if hasattr(self.init, "__array__"):
            medoids = np.hstack([np.where((X == c).all(axis=1)) for c in self.init]).ravel()
        elif self.init == "random":
            medoids = random_state_.choice(len(D), n_clusters, replace=False)
        elif self.init == "k-medoids++":
            medoids = self._kpp_init(D, n_clusters, random_state_)
        else:
            raise
        return medoids

    def _kpp_init(self, D, n_clusters, random_state_, n_local_trials=None):
        n_samples, _ = D.shape
        centers = np.empty(n_clusters, dtype=int)

        if n_local_trials is None:
            n_local_trials = 2 + int(np.log(n_clusters))

        center_id = random_state_.randint(n_samples)
        centers[0] = center_id
        closest_dist_sq = D[centers[0], :] ** 2
        current_pot = closest_dist_sq.sum()

        for cluster_index in range(1, n_clusters):
            rand_vals = (random_state_.random_sample(n_local_trials) * current_pot)
            candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), rand_vals)
            distance_to_candidates = D[candidate_ids, :] ** 2
            best_candidate = None
            best_pot = None
            best_dist_sq = None

            for trial in range(n_local_trials):
                new_dist_sq = np.minimum(closest_dist_sq, distance_to_candidates[trial])
                new_pot = new_dist_sq.sum()

                if (best_candidate is None) or (new_pot < best_pot):
                    best_candidate = candidate_ids[trial]
                    best_pot = new_pot
                    best_dist_sq = new_dist_sq

            centers[cluster_index] = best_candidate
            current_pot = best_pot
            closest_dist_sq = best_dist_sq

        return centers