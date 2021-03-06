import math

import numpy as np
from numpy.lib.stride_tricks import as_strided
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from einops import rearrange

from projections import Projections
from evaluation import get_projection_methods, evaluate
from world_generator import _tile_array, as_grid

    
def upsample_average(grid, scale_factor=2):
    upsampled = _tile_array(grid, scale_factor, scale_factor)
    upsampled /= (scale_factor * scale_factor)
    return upsampled


def plot_grid(grid_downsampled, grid_original, ax=None):
    downsampled_size = (grid_downsampled.shape[0] * grid_downsampled.shape[1])
    original_size = (grid_original.shape[0] * grid_original.shape[1])
    downsample_factor = original_size / downsampled_size
    
    # Matplotlib solution:
    if not ax:
        fig, ax = plt.subplots()

    ax.matshow(np.log(grid_downsampled / downsample_factor + 0.01) / np.log(5.0), cmap=plt.cm.Blues, vmin=0.0, vmax=3.0)

    for i in range(grid_downsampled.shape[1]):
        for j in range(grid_downsampled.shape[0]):
            c = grid_downsampled[j, i]
            # ax.text(i, j, f'{c:.1f}', va='center', ha='center')

    # Seaborn / Pandas / Jupyter solution:
    '''
    cm = sns.light_palette("blue", as_cmap=True)
    x = pd.DataFrame(intersection_matrix)
    x = x.style.background_gradient(cmap=cm)
    display(x)
    '''
    
def animate_grids(grids_downsampled_set_rows, grid_original):
    # Matplotlib solution:
    num_rows = len(grids_downsampled_set_rows)
    num_cols = max([len(row) for row in grids_downsampled_set_rows])
    fig, axs = plt.subplots(num_rows, num_cols)

    plots = []
    grids_to_plot_set = []
    
    for row_num, grid_series_row in enumerate(grids_downsampled_set_rows):
        for col_num, grid_series in enumerate(grid_series_row):
            downsampled_size = (grid_series[0].shape[0] * grid_series[0].shape[1])
            original_size = (grid_original.shape[0] * grid_original.shape[1])
            downsample_factor = original_size / downsampled_size

            grids_to_plot = [np.log(grid_downsampled / downsample_factor + 0.01) / np.log(5.0) for grid_downsampled in grid_series]
            grids_to_plot_set.append(grids_to_plot)
        
            plot = axs[row_num, col_num].matshow(grids_to_plot[0], cmap=plt.cm.Blues, vmin=0.0, vmax=3.0)
            plots.append(plot)

    def animate(t):
        for i, grids_time_series in enumerate(grids_to_plot_set):
            plots[i].set_array(grids_time_series[t])

    anim = FuncAnimation(
        fig, animate, interval=250, frames=len(grids_to_plot)-1)

    return anim


def animate_results(results):
    grid_N_a = [as_grid(N_a_t) for N_a_t in results['N_a']]

    grid_N_bs = [[as_grid(N_b_t) for N_b_t in N_bs] for N_bs in results['N_bs']]
    grid_N_b_hats = [[as_grid(N_b_t) for N_b_t in N_bs] for N_bs in results['N_b_hats']]

    anim_all_scales = animate_grids([grid_N_bs, grid_N_b_hats], grid_N_a[0])
    
    # anim_fine_grain = animate_grids(grid_N_a, grid_N_a[0])
    # anim_coarse_grain = animate_grids(grid_N_b, grid_N_a[0])
    
    return anim_all_scales
    # anim_fine_grain, anim_coarse_grain
                                                

def plot_case_counts_two_scales(N_a_time_series, N_b_time_series, coarse_grain_region, g_ba):
    t = list(range(len(N_a_time_series)))
    plt.plot(t, [N_b_t[coarse_grain_region] for N_b_t in N_b_time_series])
    fine_grain_regions = g_ba[coarse_grain_region, :].nonzero()[0]
    for fine_grain_region in fine_grain_regions:
        plt.plot(t, [N_a_t[fine_grain_region] for N_a_t in N_a_time_series])
    plt.show()


def plot_case_counts_one_scale(N_b_time_series, N_b_hat_time_series, region):
    t = list(range(len(N_b_time_series)))
    plt.plot(t, [N_b_t[region] for N_b_t in N_b_time_series])
    plt.plot(t, [N_b_hat_t[region] for N_b_hat_t in N_b_hat_time_series])
    plt.show()
