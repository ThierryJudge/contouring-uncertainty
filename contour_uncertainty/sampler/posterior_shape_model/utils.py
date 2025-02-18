import numpy as np


def index_to_flat(indices):
    """ Get list of indices of flattened list of points.

        ex: [0] for (K, 2) -> [0, 1] for (2K,)

    Args:
        indices: List of indices to flatten

    Returns:
        Flattened list of indices.
    """

    def _index_to_flat(idx):
        return [idx * 2, idx * 2 + 1]

    if isinstance(indices, int):
        return _index_to_flat(indices)
    flat_indices = []
    for idx in indices:
        flat_indices.extend(_index_to_flat(idx))

    return flat_indices


def instant_index_to_full(indices, nb_per_instant=42):

    def _instant_index_to_full(index):
        return list(np.arange(nb_per_instant * index, nb_per_instant * (index+1)))

    if isinstance(indices, int):
        return _instant_index_to_full(indices)

    full_indices = []
    for idx in indices:
        full_indices.extend(_instant_index_to_full(idx))

    return full_indices