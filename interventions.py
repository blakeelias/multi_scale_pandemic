import numpy as np


OPEN = True
CLOSED = False


def change_region_connectivity(M_a, region_idx, self_connectivity_ratio, neighbor_connectivity_ratio):
    M_a = np.copy(M_a)
    M_a[region_idx, region_idx] *= self_connectivity_ratio
    num_regions = M_a.shape[0]
    for neighbor_idx in range(num_regions):
        if neighbor_idx != region_idx:
            M_a[neighbor_idx, region_idx] *= neighbor_connectivity_ratio
    return M_a
    


def lock_down(N_a, M_a, open_close_status, lock_down_threshold=20, self_connectivity_ratio=1/4.0, neighbor_connectivity_ratio=1/4.0):
    M_a = np.copy(M_a)
    num_regions = N_a.shape[0]
    for region_idx in range(num_regions):
        if N_a[region_idx] > lock_down_threshold and open_close_status[region_idx] == OPEN:
            M_a = change_region_connectivity(M_a, region_idx, self_connectivity_ratio, neighbor_connectivity_ratio)
            open_close_status[region_idx] = CLOSED
            
    return M_a
    

def re_open(N_a, M_a, open_close_status, re_open_threshold=0, self_connectivity_ratio=4.0, neighbor_connectivity_ratio=4.0):
    M_a = np.copy(M_a)
    num_regions = N_a.shape[0]
    for region_idx in range(num_regions):
        if N_a[region_idx] <= re_open_threshold and open_close_status[region_idx] == CLOSED:
            M_a = change_region_connectivity(M_a, region_idx, self_connectivity_ratio, neighbor_connectivity_ratio)
            open_close_status[region_idx] = OPEN
    return M_a
            
