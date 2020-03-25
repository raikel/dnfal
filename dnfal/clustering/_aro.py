import pyflann
import numpy as np
from time import time
from multiprocessing import Pool
from functools import partial


def build_index(dataset, n_neighbors):
    """ Takes a dataset, returns the "n" nearest neighbors. """

    # Initialize FLANN
    pyflann.set_distance_type(distance_type='euclidean')
    flann = pyflann.FLANN()
    params = flann.build_index(dataset, algorithm='kdtree', trees=4)
    nearest_neighbors, dists = flann.nn_index(
        dataset, n_neighbors, checks=params['checks']
    )

    return nearest_neighbors, dists


def create_neighbor_lookup(nearest_neighbors):
    """
    Key is the reference face, values are the neighbors.
    """
    nn_lookup = {}
    for i in range(nearest_neighbors.shape[0]):
        nn_lookup[i] = nearest_neighbors[i, :]

    return nn_lookup


def calculate_symmetric_dist_row(nearest_neighbors, nn_lookup, row_no):
    """ Computes the symmetric distances for one row in the matrix. """

    dist_row = np.zeros([1, nearest_neighbors.shape[1]])
    f1 = nn_lookup[row_no]

    for idx, neighbor in enumerate(f1[1:]):
        Oi = idx+1
        co_neighbor = True
        try:
            row = nn_lookup[neighbor]
            Oj = np.where(row == row_no)[0][0] + 1
        except IndexError:
            Oj = nearest_neighbors.shape[1]+1
            co_neighbor = False

        # dij
        f11 = set(f1[0:Oi])
        f21 = set(nn_lookup[neighbor])
        dij = len(f11.difference(f21))
        # dji
        f12 = set(f1)
        f22 = set(nn_lookup[neighbor][0:Oj])
        dji = len(f22.difference(f12))

        if not co_neighbor:
            dist_row[0, Oi] = 9999.0
        else:
            dist_row[0, Oi] = float(dij + dji)/min(Oi, Oj)

    return dist_row


def calculate_symmetric_dist(app_nearest_neighbors):
    """ Computes the symmetric distance matrix. """

    dist_calc_time = time()
    nn_lookup = create_neighbor_lookup(app_nearest_neighbors)
    d = np.zeros(app_nearest_neighbors.shape)
    p = Pool(processes=4)
    func = partial(calculate_symmetric_dist_row, app_nearest_neighbors, nn_lookup)
    results = p.map(func, range(app_nearest_neighbors.shape[0]))
    for row_no, row_val in enumerate(results):
        d[row_no, :] = row_val
    d_time = time()-dist_calc_time
    print("Distance calculation time : {}".format(d_time))
    return d


def aro_cluster(
    features: [list, np.ndarray],
    n_neighbors: int = 10,
    distance_thr: float = 1.0
):
    """Approximate rank-order clustering.

    Takes in the nearest neighbors matrix and outputs clusters - list of lists.
    """

    app_nearest_neighbors, dists = build_index(features, n_neighbors)
    distance_matrix = calculate_symmetric_dist(app_nearest_neighbors)

    # Clustering :
    clusters = []

    # Start with the first face :
    nodes = set(list(np.arange(0, distance_matrix.shape[0])))

    plausible_neighbors = create_plausible_neighbor_lookup(
        app_nearest_neighbors, distance_matrix, distance_thr
    )

    while nodes:
        # Get a node :
        n = nodes.pop()

        # This contains the set of connected nodes :
        group = {n}

        # Build a queue with this node in it :
        queue = [n]

        # Iterate over the queue :
        while queue:
            n = queue.pop(0)
            neighbors = plausible_neighbors[n]
            # Remove neighbors we've already visited :
            neighbors = nodes.intersection(neighbors)
            neighbors.difference_update(group)

            # Remove nodes from the global set :
            nodes.difference_update(neighbors)

            # Add the connected neighbors :
            group.update(neighbors)

            # Add the neighbors to the queue to visit them next :
            queue.extend(neighbors)
        # Add the group to the list of groups :
        clusters.append(list(group))

    return clusters


def create_plausible_neighbor_lookup(
    app_nearest_neighbors,
    distance_matrix,
    thresh
):
    """Creates plausible neighbor lookup.

    Create a dictionary where the keys are the row numbers(face numbers) and
    the values are the plausible neighbors.
    """

    n_vectors = app_nearest_neighbors.shape[0]
    plausible_neighbors = {}
    for i in range(n_vectors):
        mask = distance_matrix[i, :] <= thresh
        plausible_neighbors[i] = set(
            list(app_nearest_neighbors[i, np.where(mask)][0])
        )
    return plausible_neighbors

