# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:58:50 2023

@author: thoma
"""

import pandas as pd

# List of variables you want to keep
variables = [
    "_hit_n", "_hit_avgdr", "_hit_avgpe", "_hit_firstdr","_hit_firstdz", "_hit_firstpe", 
    "_hit_lastdr", "_hit_lastdz", "_hit_totpe", "_hit_lastpe",
    "_chit_frac", "_chit_gev", "_chit_width",
    "_pass_cherenkov", "_tof_time", "_wcn_p", "_wcn_mass","_hit_width"
]

# Read in your data (assuming it's a CSV file for this example)
data = pd.read_csv('dataset.csv')

# Filter the data
filtered_data = data[variables]

# Save the filtered data to a new file (optional)
filtered_data.to_csv('filtered_data.csv', index=False)
