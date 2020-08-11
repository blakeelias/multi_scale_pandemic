import numpy as np


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