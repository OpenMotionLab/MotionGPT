import numpy as np


def prepare_vertices(vertices, canonicalize=True):
    data = vertices
    # Swap axis (gravity=Z instead of Y)
    # data = data[..., [2, 0, 1]]

    # Make left/right correct
    # data[..., [1]] = -data[..., [1]]

    # Center the first root to the first frame
    data -= data[[0], [0], :]

    # Remove the floor
    data[..., 2] -= np.min(data[..., 2])
    return data
