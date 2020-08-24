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


def plot_grid(grid_downsampled, grid_original):
    downsampled_size = (grid_downsampled.shape[0] * grid_downsampled.shape[1])
    original_size = (grid_original.shape[0] * grid_original.shape[1])
    downsample_factor = original_size / downsampled_size
    
    # Matplotlib solution:
    fig, ax = plt.subplots()
    ax.matshow(np.log(grid_downsampled / downsample_factor + 0.01) / np.log(5.0), cmap=plt.cm.Blues, vmin=0.0, vmax=3.0)

    for i in range(grid_downsampled.shape[1]):
        for j in range(grid_downsampled.shape[0]):
            c = grid_downsampled[j, i]
            ax.text(i, j, f'{c:.1f}', va='center', ha='center')

    # Seaborn / Pandas / Jupyter solution:
    '''
    cm = sns.light_palette("blue", as_cmap=True)
    x = pd.DataFrame(intersection_matrix)
    x = x.style.background_gradient(cmap=cm)
    display(x)
    '''
