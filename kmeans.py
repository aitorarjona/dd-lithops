import numpy as np
from scipy.spatial.distance import cdist
import time
from sklearn.datasets import make_blobs

import lithopsext

import lithops

lithops.utils.setup_lithops_logger(log_level='DEBUG')

N_POINTS = 200
N_DIM = 2
K = 4


def find_clusters(arr, task_group, centroids, k):
    # find members of local chunk
    print(centroids)
    dists = cdist(arr, centroids, 'sqeuclidean')
    labels = dists.argmin(1)

    # get cluster membership size from all tasks
    local_sizes = np.bincount(labels, minlength=k)
    # print(local_sizes)
    cluster_sizes = task_group.all_reduce(local_sizes,
                                          lithopsext.CollectiveOPs.SUM)

    # sum of local points for each cluster
    sum_points = np.empty((k, N_DIM))
    for cluster_id in range(k):
        arr[labels == cluster_id].sum(axis=0, out=sum_points[cluster_id])

    # find center of local points
    local_centroids = sum_points / cluster_sizes[:, None]

    # sum centers of all tasks to update centroids
    new_centroids = task_group.all_reduce(local_centroids,
                                          lithopsext.CollectiveOPs.SUM)

    return labels, new_centroids


def kmeans(X, k, iterations):
    np.random.seed(42)
    labels, centroids = np.zeros(N_DIM), np.random.rand(k, N_DIM)

    for iter_count in range(iterations):
        labels, centroids = X.parallel_apply(find_clusters,
                                             centroids,
                                             k,
                                             flatten=lambda res: (np.concatenate([tup[0] for tup in res]), res[0][1]))
        time.sleep(0.01)  # debug
        print('iter {} done'.format(iter_count))
    print('done')

    return labels, centroids


if __name__ == '__main__':
    # create dataset
    X, _ = make_blobs(n_samples=N_POINTS,
                      n_features=N_DIM,
                      centers=K,
                      cluster_std=0.5,
                      shuffle=True,
                      random_state=42)

    with lithopsext.PartitionedArray.from_numpy(X, partitions=2) as pX:
        labels, centroids = kmeans(pX, k=4, iterations=5)
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
