"""
Unsupervised Learning: Segment, Detect, Propagate with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - blobs_data
from sklearn.datasets import make_blobs

def blobs_data(random_state=42):
    # TODO: make_blobs with the five book centers and cluster_std [0.4, 0.3, 0.1, 0.1, 0.1]; return (X, y).
    centers = [[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8], [-2.8, 2.8], [-2.8, 1.3]]
    X, y = make_blobs(
        n_samples=2000,
        centers=centers,
        cluster_std=[0.4, 0.3, 0.1, 0.1, 0.1],
        random_state=random_state,
    )
    return X, y

# Step 2 - fit_kmeans
import numpy as np

class _NumpyKMeans:
    def __init__(self, n_clusters, n_init=10, random_state=42, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state
        self.max_iter = max_iter

    def _initialize(self, X, rng):
        centers = [X[rng.randint(len(X))]]
        closest_sq = np.sum((X - centers[0]) ** 2, axis=1)

        for _ in range(1, self.n_clusters):
            total = closest_sq.sum()
            if total == 0:
                idx = rng.randint(len(X))
            else:
                idx = rng.choice(len(X), p=closest_sq / total)
            centers.append(X[idx])
            closest_sq = np.minimum(
                closest_sq, np.sum((X - X[idx]) ** 2, axis=1)
            )

        return np.array(centers, dtype=float)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.RandomState(self.random_state)
        best_inertia = np.inf

        for _ in range(self.n_init):
            centers = self._initialize(X, rng)

            for _ in range(self.max_iter):
                distances = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
                labels = distances.argmin(axis=1)
                new_centers = centers.copy()

                for j in range(self.n_clusters):
                    points = X[labels == j]
                    if len(points):
                        new_centers[j] = points.mean(axis=0)
                    else:
                        farthest = distances.min(axis=1).argmax()
                        new_centers[j] = X[farthest]

                if np.allclose(new_centers, centers):
                    centers = new_centers
                    break
                centers = new_centers

            distances = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = distances.argmin(axis=1)
            inertia = distances[np.arange(len(X)), labels].sum()

            if inertia < best_inertia:
                best_inertia = inertia
                self.cluster_centers_ = centers.copy()
                self.labels_ = labels.copy()
                self.inertia_ = float(inertia)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        distances = np.sum(
            (X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2,
            axis=2,
        )
        return distances.argmin(axis=1)

def fit_kmeans(X, k, random_state=42):
    return _NumpyKMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state,
    ).fit(X)

# Step 3 - inertia_curve
def inertia_curve(X, ks):
    # TODO: {k: inertia of fit_kmeans(X, k)}.
    return {k: float(fit_kmeans(X, k).inertia_) for k in ks}

# Step 4 - silhouette_curve (not yet solved)
# TODO: implement

# Step 5 - fit_dbscan (not yet solved)
# TODO: implement

# Step 6 - dbscan_predict (not yet solved)
# TODO: implement

# Step 7 - fit_gmm (not yet solved)
# TODO: implement

# Step 8 - flag_anomalies (not yet solved)
# TODO: implement

# Step 9 - digits_data (not yet solved)
# TODO: implement

# Step 10 - baseline_50_random (not yet solved)
# TODO: implement

# Step 11 - representative_digits (not yet solved)
# TODO: implement

# Step 12 - train_on_representatives (not yet solved)
# TODO: implement

# Step 13 - propagate_and_train (not yet solved)
# TODO: implement

# Step 14 - synthetic_image (not yet solved)
# TODO: implement

# Step 15 - segment_colors (not yet solved)
# TODO: implement

# Step 16 - save_and_reload_clusterer (not yet solved)
# TODO: implement

# Step 17 - predict_digit_labels (not yet solved)
# TODO: implement

