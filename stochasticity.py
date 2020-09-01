import numpy as np
from scipy.stats import poisson
from scipy.stats import nbinom

def negative_binomial(N, p):
    "Input a vector (number of cases for each region) and return a vector that applies the negative binomial process"
    "as p reaches 0, the distribution become Poisson and the variance becomes smaller"
    stochastic = list()
    for region in N:
        r = region*(1-p)/p
        x = np.random.rand()
        cdf, k = 0, 0
        while cdf < 1:
            cdf = nbinom.cdf(k, r, 1-p)
            if x < cdf:
                stochastic.append(k)
                break
            k += 1
    return np.array(stochastic)

def poisson_process(N):
    "Input a vector (number of cases for each region) and return a vector that applies the Poisson process"
    stochastic = list()
    for region in N:
        x = np.random.rand()
        cdf, k = 0, 0
        while cdf < 1:
            cdf = poisson.cdf(k, region)
            if x < cdf:
                stochastic.append(k)
                break
            k += 1
    return np.array(stochastic)
