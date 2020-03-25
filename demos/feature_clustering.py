import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# from dnfal.engine import cluster_features
from dnfal.clustering import cluster_features


def _run(n_samples):

    distance_thr = 0.3

    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]
    features, labels_true = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=0.4,
        random_state=0
    )
    features = StandardScaler().fit_transform(features)

    timestamps = np.array(range(n_samples)).reshape((-1, 1))

    print('Features set:\n')
    print(features)

    # clusters = cluster_features(
    #     features,
    #     timestamps=timestamps,
    #     distance_thr=distance_thr,
    #     timestamp_thr=-1,
    #     grouped=True
    # )

    clusters = cluster_features(
        features,
        n_neighbors=3,
        distance_thr=distance_thr
    )

    print('\nClusters:\n')
    print(clusters)

    # Number of clusters in labels, ignoring noise if present.
    clustered = [cluster for cluster in clusters if len(cluster) > 1]
    outliers = [cluster for cluster in clusters if len(cluster) == 1]
    n_clusters_ = len(clustered)
    n_noise_ = len(clusters) - n_clusters_

    print(f'\nEstimated number of clusters: {n_clusters_}')
    print(f'Estimated number of noise points: {n_noise_}')

    # #############################################################################
    # Plot results

    colors = [
        plt.cm.Spectral(each)
        for each in np.linspace(0, 1, n_clusters_ )
    ]

    for cluster, color in zip(clustered, colors):
        xy = features[cluster]
        plt.plot(
            xy[:, 0], xy[:, 1], 'o',
            markerfacecolor=tuple(color),
            markeredgecolor='k',
            markersize=14
        )

    for outlier in outliers:
        col = [0, 0, 0, 1]
        xy = features[outlier]
        plt.plot(
            xy[:, 0], xy[:, 1], 'o',
            markerfacecolor=tuple(col),
            markeredgecolor='k',
            markersize=6
        )

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_samples',
        type=int,
        required=False,
        default=50,
        help='Number of feature vectors.'
    )
    args = parser.parse_args(sys.argv[1:])

    _run(args.n_samples)