# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:12:30 2023

@author: thoma
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('filtered_data.csv')

# Define particle cuts
particle_cuts = {
    'pimu':       (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  40) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.00) & (data['_wcn_mass'] < 0.25), 
    'kaon':       (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  70) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.39) & (data['_wcn_mass'] < 0.59),
    'electron':   (data['_pass_cherenkov'] == 1) & (data['_tof_time'] >  32) & (data['_tof_time'] <  35) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000)
}

# Filter data for each particle type
pimu_data = data[particle_cuts['pimu']]
kaon_data = data[particle_cuts['kaon']]
electron_data = data[particle_cuts['electron']]

# Define the variables you are interested in - replace with your actual variables
variables = ["_hit_n", "_hit_avgdr", "_hit_avgpe", "_hit_firstdr", "_hit_firstpe", 
             "_hit_lastdr", "_hit_lastdz", "_hit_totpe", "_hit_lastpe",
             "_chit_gev", "_chit_width", "_hit_width"]

# Function to plot histograms
def plot_histograms(data1, data2, variable, label1, label2):
    plt.hist(data1[variable], bins=50, alpha=0.5, label=label1, density=True)
    plt.hist(data2[variable], bins=50, alpha=0.5, label=label2, density=True)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {variable} for {label1} and {label2}")
    plt.legend()
    plt.show()

# Plot histograms for each variable for pimu vs kaon and pimu vs electron
for variable in variables:
    plot_histograms(pimu_data, kaon_data, variable, 'Pimu', 'Kaon')
    plot_histograms(pimu_data, electron_data, variable, 'Pimu', 'Electron')
