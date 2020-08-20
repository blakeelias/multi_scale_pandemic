import numpy as np

from visualize import _tile_array


def generate_world():
    num_regions = 64
    spread_rate = 0.05
    M_a = np.ones((num_regions, num_regions)) * spread_rate + np.eye(num_regions) * (1 - spread_rate)
    # N_a = np.random.rand(num_regions) * 10
    N_a = np.zeros((64,))
    N_a[32] = 1.0
    
    return N_a, M_a


def generate_coarse_graining(N_a):
    num_regions_a = N_a.shape[0]
    num_regions_b = int(num_regions_a / 2)

    identity = np.eye(num_regions_b)
    g_ba = _tile_array(identity, 1, 2)
    return g_ba    
