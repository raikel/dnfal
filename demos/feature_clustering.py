import argparse
import sys

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from dnfal.engine import cluster_features


def _run(n_samples):

    distance_thr = 0.5

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

    labels, clusters = cluster_features(
        features,
        timestamps=timestamps,
        distance_thr=distance_thr,
        timestamp_thr=0.9,
        min_samples=2,
        grouped=True
    )

    print('\nLabels:\n')
    print(labels)
    print('\nClusters:\n')
    print(clusters)

    # db = DBSCAN(eps=distance_thr, min_samples=2).fit(features)
    # labels = db.labels_
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[labels != -1] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print(f'\nEstimated number of clusters: {n_clusters_}')
    print(f'Estimated number of noise points: {n_noise_}')

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = features[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = features[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

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