import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import norm
from lib.misc import to_dB

# First we calculate the distance distribution of d

def generate_difference_vectors(constellation, n, d_H, n_distances=10000):
    bps = constellation.bits_per_symbol()
    
    # Generate random array of 0 and 1 values, of size n_distances x n
    m = np.random.randint(2, size=(n_distances, n))
    
    m2 = m.copy()
    # For each row (each n_distances) select d_H of the integers at random and flip their values
    for i in range(n_distances):
        indices_to_flip = np.random.choice(n, d_H, replace=False)  # Select d_H unique indices
        m2[i, indices_to_flip] = 1 - m2[i, indices_to_flip]  # Flip the values at the selected indices
        
    # Pad m and m2 to a multiple of bps
    remainder = n % bps
    if remainder != 0:
        padding_size = bps - remainder

        #print(m.shape)

        m = np.pad(m, ((0, 0), (0, padding_size)), mode='constant', constant_values=0)
        m2 = np.pad(m2, ((0, 0), (0, padding_size)), mode='constant', constant_values=0)
        
    # Create binary values per symbol
    m_grouped = (m.reshape(m.shape[0], -1, bps)*(2 ** np.arange(bps))).sum(axis=2)
    m2_grouped = (m2.reshape(m.shape[0], -1, bps)*(2 ** np.arange(bps))).sum(axis=2)
    
    # Modulate into the constellations
    s_m = constellation.symbols[constellation.bits[m_grouped]]
    s_m2 = constellation.symbols[constellation.bits[m2_grouped]]
    
    # Calculate the distances
    d = s_m - s_m2
    
    return d

#    print(m.shape)
#    print(m[0])
#    print(m2[0])
#    
#    print(m_grouped.shape)
#    print(m_grouped[0])
#    print(m2_grouped[0])
#    
#    print(s_m.shape)
#    print(s_m[0])
#    print(s_m2[0])
#    
#    print(np.abs(d[0]))

def generate_distance_distributions(constellation, n, d_H, n_distances=10000):
    d = generate_difference_vectors(constellation, n, d_H, n_distances)
    
    return np.sqrt(((np.abs(d)/2)**2).sum(axis=1))

def optimal_error_rate(Pj, N0, threshold):
    std_dev = np.sqrt(N0 / 2)
    ts = threshold - np.sqrt(Pj)
    cdf_values = norm.cdf(ts, loc=0, scale=std_dev)
    error_rate = 1 - cdf_values
    return error_rate

def optimal_error_rates(Pjs, constellation, n, d_H, N0, n_distances=1):
    dist = generate_distance_distributions(constellation, n, d_H, n_distances)
    res = []
    
    for Pj in Pjs:
        pulse_Pj = Pj * n
        error_rates = optimal_error_rate(pulse_Pj, N0, dist)
        
        for error_rate in error_rates:
            res.append({
                "JSR": Pj,
                "JSR_dB": to_dB(Pj),
                "error_rate": error_rate,
                "type": "optimal"
            })
        
    return res
