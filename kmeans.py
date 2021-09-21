import time

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import make_blobs

import lithopsext
from lithopsext.datasets.array import PartitionedArray
import lithops

# lithops.utils.setup_lithops_logger(log_level='DEBUG')

N_POINTS = 200
N_DIM = 2
K = 4


def kmeans(points, k, max_iterations, compute_group):
    np.random.seed(42)

    # get random initial centroids
    labels, centroids = np.zeros(N_DIM), np.random.rand(k, N_DIM)

    for _ in range(max_iterations):
        dists = cdist(points, centroids, 'sqeuclidean')
        labels = dists.argmin(1)

        # get cluster membership size from all tasks
        cluster_sizes = np.bincount(labels, minlength=k)
        # print(cluster_sizes)
        agg_cluster_sizes = compute_group.sync(cluster_sizes,
                                               reducer=lambda x, accum: x + accum,
                                               initial_value=np.zeros(cluster_sizes.shape))
        # print(agg_cluster_sizes)

        # time.sleep(3)

        cluster_sizes = agg_cluster_sizes

        # sum of local points for each cluster
        sum_points = np.empty((k, N_DIM))
        for cluster_id in range(k):
            points[labels == cluster_id].sum(axis=0, out=sum_points[cluster_id])

        # find center of local points
        local_centroids = sum_points / cluster_sizes[:, None]

        # sum centers of all tasks to update centroids
        # agg_centroids = compute_group.sync(local_centroids, gatherer=lambda partitions: np.add(*partitions))
        agg_centroids = compute_group.sync(local_centroids,
                                           reducer=lambda x, accum: x + accum,
                                           initial_value=np.zeros(local_centroids.shape))
        # print(agg_centroids)
        # time.sleep(3)
        centroids = agg_centroids

    print('end iteration')

    return labels, centroids


if __name__ == '__main__':
    np.random.seed(42)

    # create dataset
    X, _ = make_blobs(n_samples=N_POINTS,
                      n_features=N_DIM,
                      centers=K,
                      cluster_std=0.5,
                      shuffle=True,
                      random_state=42)

    distX = PartitionedArray.from_numpy(X, partitions=4)
    labels, centroids = distX.parallel_apply(kmeans, k=4, max_iterations=5)

    # labels, centroids = kmeans(X, k=4, max_iterations=5)
    print(centroids)

    # Compare results with scikitlearn Kmeans
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=4,
                init='random',
                n_init=10,
                max_iter=5,
                tol=1e-04,
                random_state=0)
    y_km = km.fit_predict(X)
    print(km.cluster_centers_)
