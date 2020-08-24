import math

import numpy as np
from numpy.lib.stride_tricks import as_strided
import seaborn as sns
import matplotlib.pyplot as plt
from einops import rearrange

from projections import Projections
from evaluation import get_projection_methods, evaluate
from world_generator import _tile_array, as_grid

    
def upsample_average(grid, scale_factor=2):
    upsampled = _tile_array(grid, scale_factor, scale_factor)
    upsampled /= (scale_factor * scale_factor)
    return upsampled


def plot_grid(grid):
    # Matplotlib solution:
    fig, ax = plt.subplots()
    ax.matshow(np.log(grid + 0.01) / np.log(10.0), cmap=plt.cm.Blues, vmin=0.0, vmax=3.0)

    for i in range(grid.shape[1]):
        for j in range(grid.shape[0]):
            c = grid[j,i]
            ax.text(i, j, f'{c:.1f}', va='center', ha='center')

    # Seaborn / Pandas / Jupyter solution:
    '''
    cm = sns.light_palette("blue", as_cmap=True)
    x = pd.DataFrame(intersection_matrix)
    x = x.style.background_gradient(cmap=cm)
    display(x)
    '''
