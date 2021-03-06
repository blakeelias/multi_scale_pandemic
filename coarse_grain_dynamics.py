import numpy as np


def coarse_grain_form(M_a, g_ba, g_ba_inv):
    M_b = g_ba @ M_a @ g_ba_inv
    return M_b

def _inverse_projection_form(g_ba, V):
    g_ba_inv = V @ np.linalg.inv(g_ba @ V)
    return g_ba_inv

def coarse_grain_matrix(M_a, g_ba, V):
    '''
    M_a.shape == (a, a)
    g_ba.shape == (b, a)
    V.shape == (a, b)
    '''
    g_ba_inv = _inverse_projection_form(g_ba, V)
    M_b = coarse_grain_form(M_a, g_ba, g_ba_inv)
    return M_b