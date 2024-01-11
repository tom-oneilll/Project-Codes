# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 18:12:21 2023

@author: thoma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the data
data = pd.read_csv('filtered_data.csv')

# Define variables for the BDT
variables = [
    "_hit_n", "_hit_avgdr", "_hit_avgpe", "_hit_firstdr", "_hit_firstpe", 
    "_hit_lastdr", "_hit_lastdz", "_hit_totpe", "_hit_lastpe",
    "_chit_frac", "_chit_gev", "_chit_width"
]

particle_cuts = {
    'proton':     (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  80) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.75) & (data['_wcn_mass'] < 1.13),
    'pimu':       (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  40) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.00) & (data['_wcn_mass'] < 0.25), 
    'kaon':       (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  70) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.39) & (data['_wcn_mass'] < 0.59),
    'electron':   (data['_pass_cherenkov'] == 1) & (data['_tof_time'] >  32) & (data['_tof_time'] <  35) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000)
}
# List of particle types
particles = ['proton', 'pimu']

# Number of bins for histogram
n_bins = 50

# Define upper and lower bounds for each variable
bounds = {
    "_hit_n": (0, 250),
    "_hit_avgdr": (0, 100),
    "_hit_avgpe": (0, 1000),
    "_hit_firstdr": (0, 140),
    "_hit_firstpe": (0, 1000),
    "_hit_lastdr": (0, 140),
    "_hit_lastdz": (0, 450),
    "_hit_totpe": (0, 40000),
    "_hit_lastpe": (0, 2000),
    "_chit_frac": (0.92, 1),
    "_chit_gev": (0, 2),
    "_chit_width": (0, 100)
}

# Loop through each variable
for var in variables:
    plt.figure(figsize=(10, 8))
    
    # Determine the bounds for the current variable
    lower_bound, upper_bound = bounds[var]
    
    # Create a histogram for each particle type
    for particle in particles:
        subset = data[particle_cuts[particle]]
        # Apply the bounds for the variable
        bounded_subset = subset[(subset[var] >= lower_bound) & (subset[var] <= upper_bound)]
        
        # Normalize the histogram
        weights = np.ones_like(bounded_subset[var]) / len(bounded_subset[var])
        
        plt.hist(bounded_subset[var], bins=n_bins, weights=weights, alpha=0.5, label=f'{particle} {var}')
    
    # Add legend and labels
    plt.title(f'Normalized Distribution of {var} for different particles')
    plt.xlabel(var)
    plt.ylabel('Probability')
    plt.legend(loc='upper right')
    
    # Save the figure for each variable
    plt.savefig(f'normalized_histogram_{var}.png')
    plt.close()  # Close the figure to avoid displaying it inline if not desired
