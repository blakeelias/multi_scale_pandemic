import math

import numpy as np
from numpy.lib.stride_tricks import as_strided
import seaborn as sns
import matplotlib.pyplot as plt
from einops import rearrange

from projections import Projections
from evaluation import get_projection_methods, evaluate


def as_grid(N_a):
    num_regions = N_a.shape[0]
    side_length = int(math.sqrt(num_regions))

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

    
def upsample_average(grid, scale_factor=2):
    upsampled = _tile_array(grid, scale_factor, scale_factor)
    upsampled /= (scale_factor * scale_factor)
    return upsampled


def plot_grid(grid):
    # Matplotlib solution:
    fig, ax = plt.subplots()
    ax.matshow(grid, cmap=plt.cm.Blues)

    for i in range(grid.shape[1]):
        for j in range(grid.shape[0]):
            c = grid[j,i]
            ax.text(i, j, f'{c:.2f}', va='center', ha='center')

    # Seaborn / Pandas / Jupyter solution:
    '''
    cm = sns.light_palette("blue", as_cmap=True)
    x = pd.DataFrame(intersection_matrix)
    x = x.style.background_gradient(cmap=cm)
    display(x)
    '''
