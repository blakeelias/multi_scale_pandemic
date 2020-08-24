import math

import numpy as np
from einops import rearrange


def generate_world():
    num_regions = 16
    spread_rate = 0.05
    self_spread_rate = 1.05

    M_a = np.eye(num_regions)
    N_a = np.zeros((num_regions,))

    g_ba = generate_coarse_graining(N_a)
    
    N_a_grid = as_grid(N_a)
    for i in range(N_a_grid.shape[0]):
        for j in range(N_a_grid.shape[1]):
            neighbors = neighbors_3(N_a_grid, i, j)
            for (k, l) in neighbors:
                region_idx_self = row_col_to_index(N_a_grid, i, j)
                region_idx_neighbor = row_col_to_index(N_a_grid, k, l)
                M_a[region_idx_self, region_idx_neighbor] = spread_rate
                M_a[region_idx_neighbor, region_idx_self] = spread_rate

    for i in range(N_a.shape[0]):
        M_a[i, i] = self_spread_rate
                
    return N_a, M_a, g_ba


def as_grid(N_a):
    # print('in as_grid')
    num_regions = N_a.shape[0]
    # print('num_regions', num_regions)
    # print('N_a.shape', N_a.shape)
    side_length = int(math.sqrt(num_regions))
    # print('side_length', side_length)
    
    return rearrange(N_a, '(a b) -> a b', a=side_length)


def _tile_array(a, b0, b1):
    kernel = np.ones((b0, b1), a.dtype)
    return np.kron(a, kernel)

    '''
    # Alternative (faster) -- https://stackoverflow.com/questions/32846846/quick-way-to-upsample-numpy-array-by-nearest-neighbor-tiling
    r, c = a.shape                                    # number of rows/columns
    rs, cs = a.strides                                # row/column strides 
    x = as_strided(a, (r, b0, c, b1), (rs, 0, cs, 0)) # view a as larger 4D array
    return x.reshape(r*b0, c*b1)                      # create new 2D array
    '''


def neighbors_3(grid, i, j):
    candidates = [
        (i, j),
        (i-1, j),
        (i+1, j),
        (i, j-1),
        (i-1, j-1),
        (i+1, j-1),
        (i, j+1),
        (i-1, j+1),
        (i+1, j+1)
    ]

    return [(i, j) for (i, j) in candidates if (i >= 0 and i < grid.shape[0]) and (j >= 0 and j < grid.shape[1])]


def neighbors_2(grid, i, j):
    candidates = [
        (i, j),
        (i+1, j),
        (i, j+1),
        (i+1, j+1)
    ]

    return [candidate for candidate in candidates if (i >= 0 and i < grid.shape[0]) and (j >= 0 and j < grid.shape[1])]
    

def row_col_to_index(grid, i, j):
    return i * grid.shape[0] + j
    

def generate_coarse_graining(N_a):
    # print('in generate_coarse_graining')
    num_regions_a = N_a.shape[0]
    # print('num_regions_a', num_regions_a)
    num_regions_b = int(num_regions_a / 4)

    # print('num_regions_a, num_regions_b')
    # print(num_regions_a, num_regions_b)

    N_b = np.zeros((num_regions_b,))
    # print('N_b.shape', N_b.shape)
    
    grid_N_a = as_grid(N_a)
    grid_N_b = as_grid(N_b)

    # print('grid_N_a.shape')
    # print(grid_N_a.shape)
    # print('grid_N_b.shape')
    # print(grid_N_b.shape)
    
    g_ba = np.zeros((num_regions_b, num_regions_a))

    
    for coarse_grain_row in range(grid_N_b.shape[0]):
        for coarse_grain_col in range(grid_N_b.shape[1]):
            coarse_grain_idx = row_col_to_index(grid_N_b, coarse_grain_row, coarse_grain_col)
            members = neighbors_2(grid_N_a, coarse_grain_row * 2, coarse_grain_col * 2)
            for (fine_grain_row, fine_grain_col) in members:
                fine_grain_idx = row_col_to_index(grid_N_a, fine_grain_row, fine_grain_col)
                g_ba[coarse_grain_idx, fine_grain_idx] = 1
    
    return g_ba    
