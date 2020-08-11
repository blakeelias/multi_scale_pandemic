import numpy as np

from coarse_grain_dynamics import coarse_grain_matrix, _coarse_grain_form


def moore_penrose(M_a, g_ba):
    g_ba_inv = np.linalg.pinv(g_ba)
    M_b = _coarse_grain_form(M_a, g_ba, g_ba_inv)
    return M_b

def top_b_eigenvectors(M_a, g_ba):
    n_b, n_a = g_ba.shape
    
    Lambda_col_a, V_a = np.linalg.eig(M_a)
    
    idx = Lambda_col_a.argsort()[-n_b:][::-1]   
    Lambda_col_a_top_b = Lambda_col_a[idx]  # top n_b eigenvalues
    V_a_top_b = V_a[:,idx]           # corresponding eigenvectors
    M_b = coarse_grain_matrix(M_a, g_ba, V_a_top_b)
    
    return M_b

def top_eigenvector(M_a, g_ba):
    n_b, n_a = g_ba.shape
    
    Lambda_col_a, V_a = np.linalg.eig(M_a)
    
    idx = Lambda_col_a.argsort()[-1:][::-1]
    Lambda_col_a_top_b = Lambda_col_a[idx]  # top 1 eigenvalue
    V_a_top_1 = V_a[:,idx]           # corresponding eigenvectors
    V = np.tile(V, n_b)
    M_b = coarse_grain_matrix(M_a, g_ba, V)
    
    return M_b

def top_eigenvector_normalized(M_a, g_ba):
    n_b, n_a = g_ba.shape
    
    Lambda_col_a, V_a = np.linalg.eig(M_a)
    
    idx = Lambda_col_a.argsort()[-1:][::-1]
    Lambda_col_a_top_b = Lambda_col_a[idx]  # top 1 eigenvalue
    top_eigenvector = V_a[:,idx]           # corresponding eigenvectors
    
    # Note: not truly an inverse, if g_ba is not all 1s/0s.
    # If we scale g_ba, we will also scale g_ba @ g_ab.
    # No issue if g_ba is all 1s / 0s.
    # Maybe can resolve this potential issue, normalizing by g_ba somehow?
    g_ab = np.diag(top_eigenvector) @ g_ba.T @ np.linalg.inv(np.diag(g_ba @ top_eigenvector))
    
    M_b = _coarse_grain_form(M_a, g_ba, g_ab)
    
    return M_b