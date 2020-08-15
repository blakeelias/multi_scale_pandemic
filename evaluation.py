from pprint import pprint
import inspect
import numpy as np

import projections


def evolve(M, N_0, num_steps=10):
    return [np.linalg.matrix_power(M, t) @ N_0 for t in range(num_steps)]

def evaluate(M_a=None, M_b=None, g_ba=None, N_a_0=None, num_steps=10):
    N_b_0 = g_ba @ N_a_0
    
    N_a = evolve(M_a, N_a_0, num_steps=num_steps)
    N_b = [g_ba @ N_a_t for N_a_t in N_a]
    N_b_hat = evolve(M_b, N_b_0, num_steps=num_steps)

    return {
        'N_a': N_a,
        'N_b': N_b,
        'N_b_hat': N_b_hat
    }

def get_projection_methods():
    projector = projections.Projections()
    methods = inspect.getmembers(projector, inspect.ismethod)
    return methods

def main():
    M_a = np.ones((4, 4)) * 0.01 + np.diag([8, 4, 1, 1])
    g_ba = np.array([
        [1., 0, 1, 0],
        [0, 1, 0, 1]
    ])
    N_a_0 = np.array([[1., 0., 1., 0.]]).T
    
    print('M_a:')
    print(M_a)
    
    print('g_ba:')
    print(g_ba)
    
    projection_methods = get_projection_methods()
    
    results = {}
    for (method_name, projection_method) in projection_methods:
        M_b = projection_method(M_a, g_ba)
        results[method_name] = evaluate(M_a, M_b, g_ba, N_a_0)

    print('=' * 80)
    print('Results:')
    pprint(results)
        
if __name__ == '__main__':
    main()