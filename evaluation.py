from pprint import pprint
import inspect
from pdb import set_trace as b

import numpy as np

import projections


def evolve(M, N_0, num_steps=10):
    return [np.linalg.matrix_power(M, t) @ N_0 for t in range(num_steps)]

def evaluate(M_a=None, projection_method=None, g_bas=None, N_a_0=None, num_steps=10):
    if not g_bas:
        g_bas = []

    if not projection_method:
        projection_method = get_projection_methods()[0]

    M_bs = []
    N_bs = []
    N_b_hats = []

    M_bs.append(M_a)
    N_a = evolve(M_a, N_a_0, num_steps=num_steps)
    N_bs.append(N_a)
    N_b_hats.append(N_a)

    for i, g_ba in enumerate(g_bas):
        g_ba_cum = np.linalg.multi_dot(g_bas[:i+1])

        M_b = projection_method(M_bs[-1], g_ba)
        N_b_0 = g_ba_cum @ N_a_0

        N_b = [g_ba @ N_b_t for N_b_t in N_bs[-1]]
        N_b_hat = evolve(M_b, N_b_0, num_steps=num_steps)

        M_bs.append(M_b)
        N_bs.append(N_b)
        N_b_hats.append(N_b_hat)

    return {
        'N_a': N_a,
        'N_bs': N_bs,
        'M_bs': M_bs,
        'N_b_hats': N_b_hats
    }

def get_projection_methods():
    projector = projections.Projections()
    methods = inspect.getmembers(projector, inspect.ismethod)
    return methods

def main():
    M_a = np.ones((4, 4)) * 0.05 + np.diag([3, 2, 1, .5])
    M_a[0, 1:] = 0.01
    M_a[1:, 0] = 0.01
    
    g_ba = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=int)
    N_a_0 = np.array([[1., 0., 1., 0.]]).T
    
    '''
    print('M_a:')
    print(M_a)
    
    print('g_ba:')
    print(g_ba)
    
    results = {}
    for (method_name, projection_method) in projection_methods:
        M_b = projection_method(M_a, g_ba)
        results[method_name] = evaluate(M_a, M_b, g_ba, N_a_0)

    print('=' * 80)
    print('Results:')
    pprint(results)
    '''
    
    projection_methods = get_projection_methods()
    
    print(M_a)
    for (method_name, projection_method) in projection_methods:
        print('=' * 80)
        M_b = projection_method(M_a, g_ba)
        print(method_name)
        print(M_b)
        #results = evaluate(M_a, M_b, g_ba, N_a_0)
        #print('Time series')
        #print(results)
        
        if method_name == 'sub_matrix_eigenvector':
            M_b, g_ab = projection_method(M_a, g_ba, return_g_ab=True)
            
            lam_coarse, v_coarse = projections.top_eigenvectors(M_b, 1) # top 1 eigenvector
            lam_fine, v_fine = projections.top_eigenvectors(M_a, 1) # top 1 eigenvector
            
            v_coarse_projected = g_ab @ v_coarse

            print('lam_coarse, v_coarse, v_coarse_projected')
            print(lam_coarse)
            print(v_coarse)
            print(v_coarse_projected)
            
            print('lam_fine, v_fine')
            print(lam_fine)
            print(v_fine)
            

        
if __name__ == '__main__':
    main()
