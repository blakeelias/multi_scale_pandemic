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
    
def animate_grids(grids_downsampled_set, grid_original):
    # Matplotlib solution:
    fig, axs = plt.subplots(1, 2)

    plots = []
    grids_to_plot_set = []
    
    for i, grid_series in enumerate(grids_downsampled_set):
        downsampled_size = (grid_series[0].shape[0] * grid_series[0].shape[1])
        original_size = (grid_original.shape[0] * grid_original.shape[1])
        downsample_factor = original_size / downsampled_size

        grids_to_plot = [np.log(grid_downsampled / downsample_factor + 0.01) / np.log(5.0) for grid_downsampled in grid_series]
        grids_to_plot_set.append(grids_to_plot)
        
        plot = axs[i].matshow(grids_to_plot[0], cmap=plt.cm.Blues, vmin=0.0, vmax=3.0)
        plots.append(plot)

    def animate(t):
        for i, grids_time_series in enumerate(grids_to_plot_set):
            plots[i].set_array(grids_time_series[t])

    anim = FuncAnimation(
        fig, animate, interval=250, frames=len(grids_to_plot)-1)

    return anim


def animate_results(results):
    grid_N_a = [as_grid(N_a_t) for N_a_t in results['N_a']]
    grid_N_b = [as_grid(N_b_t) for N_b_t in results['N_bs'][1]]

    anim_all_scales = animate_grids([grid_N_a, grid_N_b], grid_N_a[0])
    
    # anim_fine_grain = animate_grids(grid_N_a, grid_N_a[0])
    # anim_coarse_grain = animate_grids(grid_N_b, grid_N_a[0])
    
    return anim_all_scales
    # anim_fine_grain, anim_coarse_grain
                                                

def dummy_animate():
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set(xlim=(-3, 3), ylim=(-1, 1))
    
    x = np.linspace(-3, 3, 91)
    t = np.linspace(1, 25, 30)
    X2, T2 = np.meshgrid(x, t)
    
    sinT2 = np.sin(2*np.pi*T2/T2.max())
    F = 0.9*sinT2*np.sinc(X2*(1 + sinT2))
    
    line = ax.plot(x, F[0, :], color='k', lw=2)[0]
    
    def animate(i):
        line.set_ydata(F[i, :])

    anim = FuncAnimation(
        fig, animate, interval=100, frames=len(t)-1)

    return anim
