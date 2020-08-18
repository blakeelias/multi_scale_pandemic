from pdb import set_trace as b

import numpy as np

from coarse_grain_dynamics import coarse_grain_form, coarse_grain_matrix



def top_eigenvectors(M, k):
    Lambda, V = np.linalg.eig(M)

    idx = Lambda.argsort()[-k:][::-1]   
    Lambda_top_k = Lambda[idx]  # top n_b eigenvalues
    V_top_k = V[:,idx]           # corresponding eigenvectors

    return Lambda_top_k, V_top_k
    
    
class Projections:
    def moore_penrose(self, M_a, g_ba):
        '''
        Use Moore-Penrose pseudo-inverse as the g_ab matrix (right-inverse to g_ba).
        Divides cases in a coarse-grained region equally into all fine-grained regions it is comprised of.
        '''
        g_ba_inv = np.linalg.pinv(g_ba)
        M_b = coarse_grain_form(M_a, g_ba, g_ba_inv)
        return M_b

    def top_b_eigenvectors(self, M_a, g_ba):
        '''
        Use top b eigenvectors of M_a as the `V` matrix.
        '''
        n_b, n_a = g_ba.shape

        Lambda_col_a, V_a = np.linalg.eig(M_a)

        idx = Lambda_col_a.argsort()[-n_b:][::-1]   
        Lambda_col_a_top_b = Lambda_col_a[idx]  # top n_b eigenvalues
        V_a_top_b = V_a[:,idx]           # corresponding eigenvectors
        M_b = coarse_grain_matrix(M_a, g_ba, V_a_top_b)

        return M_b

    '''def top_eigenvector(self, M_a, g_ba):
        n_b, n_a = g_ba.shape

        Lambda_col_a, V_a = np.linalg.eig(M_a)

        idx = Lambda_col_a.argsort()[-1:][::-1]
        Lambda_col_a_top_b = Lambda_col_a[idx]  # top 1 eigenvalue
        V_a_top_1 = V_a[:,idx]           # corresponding eigenvectors
        V = np.tile(V_a_top_1, n_b)
        M_b = coarse_grain_matrix(M_a, g_ba, V)

        return M_b
    '''

    def top_eigenvector_normalized(self, M_a, g_ba):
        '''
        Assign the cases in a coarse-grained region, only to the fine-grained regions it is composed of.
        Divide the cases according to their ratio in the top eigenvector.
        '''
        n_b, n_a = g_ba.shape

        Lambda_col_a, V_a = np.linalg.eig(M_a)

        idx = Lambda_col_a.argsort()[-1:][::-1]
        Lambda_col_a_top_b = Lambda_col_a[idx]  # top 1 eigenvalue
        top_eigenvector_ = V_a[:,idx]           # corresponding eigenvectors

        # Note: not truly an inverse, if g_ba is not all 1s/0s.
        # If we scale g_ba, we will also scale g_ba @ g_ab.
        # No issue if g_ba is all 1s / 0s.
        # Maybe can resolve this potential issue, normalizing by g_ba somehow?
        top_eigenvector = top_eigenvector_.reshape((top_eigenvector_.shape[0],))
        projected_eigenvector_ = g_ba @ top_eigenvector
        projected_eigenvector = projected_eigenvector_.reshape((projected_eigenvector_.shape[0],))
        
        g_ab = np.diag(top_eigenvector) @ g_ba.T @ np.linalg.inv(np.diag(projected_eigenvector))

        M_b = coarse_grain_form(M_a, g_ba, g_ab)

        return M_b
    
    def sub_matrix_eigenvector(self, M_a, g_ba, return_g_ab=False):
        '''
        Evaluate the sub-matrices of M_a corresponding to the regions grouped by g_ba.
        Take top eigenvector of each sub-matrix, divide cases according to that eigenvector.
        '''
        n_b, n_a = g_ba.shape
        
        g_ab = g_ba.T * 1.0   # turn int to float
        
        for i in range(n_b):
            row_selector = g_ba[i:i+1, :]
            col_selector = g_ba[i:i+1, :].T
            
            sub_matrix = row_selector * M_a * col_selector
            
            lam, v = np.linalg.eig(sub_matrix)
            eig_idx = lam.argsort()[-1:][::-1]
            top_lambda = lam[eig_idx]    # top eigenvalue
            top_eigenvector = v[:, eig_idx]  # corresponding eigenvector
            
            g_ab[:, i:i+1] *= top_eigenvector / sum(top_eigenvector)
        
        M_b = coarse_grain_form(M_a, g_ba, g_ab)
        if return_g_ab:
            return M_b, g_ab
        return M_b
        