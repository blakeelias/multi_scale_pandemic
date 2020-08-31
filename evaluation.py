from pprint import pprint
import inspect
from pdb import set_trace as b

import numpy as np

import projections
import visualize
import interventions


def evolve(M, N_0, num_steps=10, lock_down_threshold=1e6, re_open_threshold=0, intervention_strategy=True):
    M_current = M
    M_effective = M_current
    num_regions = N_0.shape[0]
    
    open_closed_status = [interventions.OPEN] * num_regions
    
    N_t = []
    for t_i in range(num_steps):
        N_t_i = M_effective @ N_0
        N_t.append(N_t_i)

        if intervention_strategy:
            M_current = interventions.lock_down(N_t_i, M_current, open_closed_status, lock_down_threshold=lock_down_threshold)
            M_current = interventions.re_open(N_t_i, M_current, open_closed_status, re_open_threshold=re_open_threshold)
            
        M_effective = M_current @ M_effective
    
    return N_t # [np.linalg.matrix_power(M, t) @ N_0 for t in range(num_steps)]


def evaluate(M_a=None, projection_method=None, g_bas=None, N_a_0=None, num_steps=10, lock_down_threshold=1e10, re_open_threshold=-1, intervention_strategy=True):    
    if not g_bas:
        g_bas = []

    if not projection_method:
        method_name, projection_method = get_projection_methods()[0]

    M_bs = []
    N_bs = []
    N_b_hats = []

    M_bs.append(M_a)
    N_a = evolve(M_a, N_a_0, num_steps=num_steps, lock_down_threshold=lock_down_threshold, re_open_threshold=re_open_threshold, intervention_strategy=intervention_strategy)
    N_bs.append(N_a)
    N_b_hats.append(N_a)

    for i, g_ba in enumerate(g_bas):
        if i == 0:
            g_ba_cum = g_ba
        else:
            g_ba_cum = np.linalg.multi_dot(list(reversed(g_bas[:i+1])))

        M_b = projection_method(M_bs[-1], g_ba)
        N_b_0 = g_ba_cum @ N_a_0

        N_b = [g_ba @ N_b_t for N_b_t in N_bs[-1]]

        ratio = g_ba.shape[1] / g_ba.shape[0]
        scaled_lock_down_threshold = lock_down_threshold * ratio
        scaled_re_open_threshold = re_open_threshold * ratio
        
        N_b_hat = evolve(M_b, N_b_0, num_steps=num_steps, lock_down_threshold=scaled_lock_down_threshold, re_open_threshold=scaled_re_open_threshold, intervention_strategy=intervention_strategy)

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
