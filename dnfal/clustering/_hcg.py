from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, OPTICS
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering
import networkx as nx


def cluster_labels(labels, indexes):
    labels_set = set(labels) - {-1}

    clusters = {label: [] for label in labels_set}
    outliers = []

    for index, label in zip(indexes, labels):
        if label != -1:
            clusters[label].append(index)
        else:
            outliers.append([index])

    return list(clusters.values()) + outliers


def hcg_cluster(
    features,
    timestamps=None,
    linkage='ward',
    distance_thr: tuple = (0.5, 0.5),
    timestamp_thr: float = 0,
    edge_thr: float = 0.7
):
    dist_neigh = NearestNeighbors(radius=distance_thr[0])
    dist_neigh.fit(features)
    dist_graph = dist_neigh.radius_neighbors_graph(mode='connectivity')

    if timestamps is not None and timestamp_thr > 0:
        time_neigh = NearestNeighbors(radius=timestamp_thr)
        time_neigh.fit(timestamps)
        time_graph = time_neigh.radius_neighbors_graph(mode='connectivity')
        dist_graph = dist_graph.multiply(time_graph)
        dist_graph.eliminate_zeros()

    dist_graph = nx.from_scipy_sparse_matrix(dist_graph)

    components = nx.connected_components(dist_graph)

    clusters = []

    clustering = None
    if distance_thr[1] > 0:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_thr[1],
            affinity='euclidean',
            linkage=linkage
        )

    for component in components:
        n = len(component)
        if n > 3 and clustering is not None:
            sub_graph = dist_graph.subgraph(component).copy()
            component = list(component)
            if sub_graph.size() >= edge_thr * (n * (n - 1) / 2):
                clusters.append(component)
            else:
                clustering.fit(features[component])
                clusters.extend(cluster_labels(clustering.labels_, component))
        else:
            clusters.append(list(component))

    return clusters


# def cluster_features(
#     features,
#     timestamps=None,
#     distance_thr: float = 0.5,
#     timestamp_thr: float = 0,
#     grouped: bool = True
# ):
#     dist_neigh = NearestNeighbors(radius=distance_thr)
#     dist_neigh.fit(features)
#     dist_graph = dist_neigh.radius_neighbors_graph(mode='connectivity')
#
#     if timestamps is not None and timestamp_thr > 0:
#         time_neigh = NearestNeighbors(radius=timestamp_thr)
#         time_neigh.fit(timestamps)
#         time_graph = time_neigh.radius_neighbors_graph(mode='connectivity')
#         dist_graph = dist_graph.multiply(time_graph)
#         dist_graph.eliminate_zeros()
#
#     clustering = AgglomerativeClustering(
#         n_clusters=None,
#         distance_threshold=distance_thr,
#         affinity='euclidean',
#         linkage='ward',
#         connectivity=dist_graph
#     )
#
#     clustering.fit(features)
#
#     labels = clustering.labels_
#
#     if not grouped:
#         return labels, None
#
#     labels_set = set(labels) - {-1}
#
#     clusters = {label: [] for label in labels_set}
#     outliers = []
#
#     for ind, label in enumerate(labels):
#         if label != -1:
#             clusters[label].append(ind)
#         else:
#             outliers.append([ind])
#
#     return labels, list(clusters.values()) + outliers

# import networkx as nx
# from .clustering import cluster_features as aro_cluster
#
# def cluster_features(
#     features,
#     timestamps=None,
#     distance_thr: float = 0.5,
#     timestamp_thr: float = 0,
#     grouped: bool = True
# ):
#     n_features = len(features)
#     dist_neigh = NearestNeighbors(radius=1.13)
#     dist_neigh.fit(features)
#     dist_graph = dist_neigh.radius_neighbors_graph(mode='connectivity')
#
#     if timestamps is not None and timestamp_thr > 0:
#         time_neigh = NearestNeighbors(radius=timestamp_thr)
#         time_neigh.fit(timestamps)
#         time_graph = time_neigh.radius_neighbors_graph(mode='connectivity')
#         dist_graph = dist_graph.multiply(time_graph)
#         dist_graph.eliminate_zeros()
#
#     dist_graph = nx.from_scipy_sparse_matrix(dist_graph)
#
#     components = nx.connected_components(dist_graph)
#
#     clusters = []
#
#     for component in components:
#         clusters.append(list(component))
#         n = len(component)
#         if n > 3:
#             sub_graph = dist_graph.subgraph(component).copy()
#             component = list(component)
#             if sub_graph.size() == (n * (n - 1) / 2):
#                 clusters.append(component)
#             else:
#                 sub_clusters = aro_cluster(
#                     features[component],
#                     distance_thr=distance_thr,
#                     n_neighbors=min(n, 10)
#                 )
#                 for sub_cluster in sub_clusters:
#                     clusters.append([component[i] for i in sub_cluster])
#         else:
#             clusters.append(list(component))
#
#     return clusters


# def cluster_features(
#     features,
#     timestamps=None,
#     distance_thr: float = 0.5,
#     timestamp_thr: float = 0,
#     grouped: bool = True
# ):
#     n_features = len(features)
#     dist_neigh = NearestNeighbors(radius=distance_thr)
#     dist_neigh.fit(features)
#     dist_graph = dist_neigh.radius_neighbors_graph(mode='connectivity')
#
#     if timestamps is not None and timestamp_thr > 0:
#         time_neigh = NearestNeighbors(radius=timestamp_thr)
#         time_neigh.fit(timestamps)
#         time_graph = time_neigh.radius_neighbors_graph(mode='connectivity')
#         dist_graph = dist_graph.multiply(time_graph)
#         dist_graph.eliminate_zeros()
#
#     _, labels = connected_components(
#         csgraph=dist_graph,
#         directed=False,
#         return_labels=True
#     )
#
#     clusters_high = cluster_labels(labels, range(0, n_features))
#
#     # Second level clustering
#     clusters_low = []
#     clustering = AgglomerativeClustering(
#         n_clusters=None,
#         distance_threshold=1.2,
#         affinity='euclidean',
#         linkage='single'
#     )
#
#     # clustering = OPTICS(
#     #     max_eps=distance_thr,
#     #     min_samples=3,
#     #     metric='euclidean'
#     # )
#
#     for cluster in clusters_high:
#         if len(cluster) > 3:
#             clustering.fit(features[cluster])
#             clusters_low.extend(
#                 cluster_labels(clustering.labels_, cluster)
#             )
#         else:
#             clusters_low.append(cluster)
#
#     return clusters_low


# import networkx as nx
#
# def cluster_features(
#     features,
#     timestamps=None,
#     distance_thr: float = 0.5,
#     timestamp_thr: float = 0,
#     grouped: bool = True
# ):
#     n_features = len(features)
#     dist_neigh = NearestNeighbors(radius=distance_thr)
#     dist_neigh.fit(features)
#     dist_graph = dist_neigh.radius_neighbors_graph(mode='connectivity')
#
#     if timestamps is not None and timestamp_thr > 0:
#         time_neigh = NearestNeighbors(radius=timestamp_thr)
#         time_neigh.fit(timestamps)
#         time_graph = time_neigh.radius_neighbors_graph(mode='connectivity')
#         dist_graph = dist_graph.multiply(time_graph)
#         dist_graph.eliminate_zeros()
#
#     dist_graph = nx.from_scipy_sparse_matrix(dist_graph)
#
#     components = nx.connected_components(dist_graph)
#
#     clusters = []
#
#     for component in components:
#         if len(component) > 3:
#             sub_graph = dist_graph.subgraph(component).copy()
#             sub_components = nx.k_edge_components(sub_graph, k=2)
#             clusters.extend([
#                 list(sub_component) for sub_component in sub_components
#             ])
#         else:
#             clusters.append(list(component))
#
#     return clusters