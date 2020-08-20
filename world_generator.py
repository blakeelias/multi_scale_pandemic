import numpy as np


def generate_world():
    M_a = np.eye(64)
    N_a = np.random.rand(64) * 10
    # N_a = np.zeros((64,))

    return N_a, M_a
