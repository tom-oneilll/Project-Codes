# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 04:13:27 2023

@author: thoma
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('dataset.csv')

# Define particle cuts
particle_cuts = {
    'proton':     (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  80) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.75) & (data['_wcn_mass'] < 1.13),
    'pimu':       (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  40) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.00) & (data['_wcn_mass'] < 0.25), 
    'kaon':       (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  70) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.39) & (data['_wcn_mass'] < 0.59),
    'electron':   (data['_pass_cherenkov'] == 1) & (data['_tof_time'] >  32) & (data['_tof_time'] <  35) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000)
}



# Print the total count of the entire data
print(f"Total count of the entire data: {data.shape[0]}")

# Count occurrences for each particle type
particle_counts = {}
for particle, cut in particle_cuts.items():
    particle_counts[particle] = data[cut].shape[0]

# Display the counts
for particle, count in particle_counts.items():
    print(f"{particle.capitalize()}: {count}")

# Print total identified counts
total_identified = sum(particle_counts.values())
print(f"Total identified counts: {total_identified}")

# Check for overlaps
overlap_dfs = {}
checked_pairs = set()
for particle, cut in particle_cuts.items():
    for other_particle, other_cut in particle_cuts.items():
        if particle != other_particle and (particle, other_particle) not in checked_pairs and (other_particle, particle) not in checked_pairs:
            overlap = data[cut & other_cut]
            overlap_name = f"{particle}_and_{other_particle}"
            overlap_dfs[overlap_name] = overlap
            print(f"There are {overlap.shape[0]} duplicates between the {particle} and {other_particle} groups.")
            checked_pairs.add((particle, other_particle))

# Create dataframes for each particle group without overlaps
particle_dfs_without_overlaps = {}
for particle, cut in particle_cuts.items():
    # Exclude overlaps with other groups
    no_overlap_cut = cut.copy()
    for other_particle, other_cut in particle_cuts.items():
        if particle != other_particle:
            no_overlap_cut &= ~other_cut
    particle_dfs_without_overlaps[particle] = data[no_overlap_cut]

# Display the counts for each particle group without overlaps
for particle, df in particle_dfs_without_overlaps.items():
    print(f"{particle.capitalize()} without overlaps: {df.shape[0]}")


# Calculate and display the total identified counts minus overlaps
total_identified_no_overlap = sum(df.shape[0] for _, df in overlap_dfs.items())
print(f"Total identified counts minus overlaps: {total_identified - total_identified_no_overlap}")

# Display the counts for overlaps between each group
for overlap_name, df in overlap_dfs.items():
    print(f"Overlap between {overlap_name}: {df.shape[0]}")




time_of_flight_column = '_tof_time'
momentum_column = '_wcn_p'
particle_pairs = [
    ('proton', 'electron'),
    ('pimu', 'electron'),
    ('kaon', 'electron')
]

for particle, other_particle in particle_pairs:
    # Get data for each particle
    particle_df = particle_dfs_without_overlaps[particle]
    other_particle_df = particle_dfs_without_overlaps[other_particle]
    overlap_df = overlap_dfs[f"{particle}_and_{other_particle}"]

    # Plot data
    plt.figure(figsize=(10, 6))
    plt.scatter(particle_df[momentum_column], particle_df[time_of_flight_column], label=particle, alpha=0.6)
    plt.scatter(other_particle_df[momentum_column], other_particle_df[time_of_flight_column], label=other_particle, alpha=0.6)
    plt.scatter(overlap_df[momentum_column], overlap_df[time_of_flight_column], label=f"{particle}_and_{other_particle} overlap", color='red', alpha=0.6)

    # Labeling the plot
    plt.title(f"{particle.capitalize()} vs {other_particle.capitalize()} with overlaps")
    plt.xlabel('Momentum (MeV/c)')
    plt.ylabel('Time of Flight (ns)')
    #plt.ylim(bottom=0)
    #plt.xlim(left=0)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
# Start a single plot
plt.figure(figsize=(10, 6))

# Loop through particle pairs and plot data points and overlaps
for particle, other_particle in particle_pairs:
    # Get data for each particle
    particle_df = particle_dfs_without_overlaps[particle]
    
    # Only get overlap data if it exists
    overlap_df = overlap_dfs.get(f"{particle}_and_{other_particle}", None)

    # Plot particle data
    plt.scatter(particle_df[momentum_column], particle_df[time_of_flight_column], label=particle, alpha=0.6)
    
    # If there's overlap data with electron, plot it
    if overlap_df is not None and "electron" in other_particle:
        plt.scatter(overlap_df[momentum_column], overlap_df[time_of_flight_column], label=f"{particle}_and_{other_particle} overlap", color='red', alpha=0.6)

# Setting the y-axis limit
plt.ylim(bottom=0)

# Labeling the plot
plt.title('Particle Distributions with electron Overlaps')
plt.xlabel('Momentum (MeV/c)')
plt.ylabel('Time of Flight (ns)')
plt.legend()
plt.grid(True)
plt.show()







particle_pairs = [
    ('proton', 'electron'),
    ('pimu', 'electron'),
    ('kaon', 'electron'),
    ('electron', None)  # Added this to plot standalone 'electron'
]


# Loop through particle pairs to create separate plots
for particle, other_particle in particle_pairs:
    # Start a new plot for each particle
    plt.figure(figsize=(10, 6))
    
    # Get data for the particle
    particle_df = particle_dfs_without_overlaps[particle]
    
    # Plot particle data
    plt.scatter(particle_df[momentum_column], particle_df[time_of_flight_column], label=particle, alpha=0.6)
    
    # If there's another particle specified (i.e., not the standalone 'electron' plot), get the overlap data
    if other_particle:
        overlap_df = overlap_dfs.get(f"{particle}_and_{other_particle}", None)
        # If there's overlap data, plot it
        if overlap_df is not None:
            plt.scatter(overlap_df[momentum_column], overlap_df[time_of_flight_column], label=f"{particle}_and_{other_particle} overlap", color='red', alpha=0.6)

    # Setting the y-axis limit
    #plt.ylim(bottom=0)

    # Labeling the plot
    if other_particle:
        plt.title(f'{particle.capitalize()} Distribution with {other_particle.capitalize()} Overlaps')
    else:
        plt.title(f'{particle.capitalize()} Distribution')
        
    plt.xlabel('Momentum (MeV/c)')
    plt.ylabel('Time of Flight (ns)')
    plt.legend()
    plt.grid(True)
    plt.show()



"""
# Extract the electron dataframe from the previously defined particle_dfs_without_overlaps
electron_df = particle_dfs_without_overlaps['electron']

# Filter out the error values and values above 100 from the electron dataframe
filtered_electron_df = electron_df[(electron_df[time_of_flight_column] > 27) & (electron_df[time_of_flight_column] < 38)]

# Start a new plot for the histogram
plt.figure(figsize=(10, 6))

# Plot histogram for the time of flight of electron after filtering
plt.hist(filtered_electron_df[time_of_flight_column], bins=500, alpha=0.7, color='blue', edgecolor='black')

# Labeling the plot
plt.title('Time of Flight Distribution for electron (Between 0 and 100)')
plt.xlabel('Time of Flight (ns)')
plt.ylabel('Count')
plt.grid(True)

plt.show()


"""

# Define a list of particle types based on keys in the particle_dfs_without_overlaps dictionary
particle_types = list(particle_dfs_without_overlaps.keys())

for particle in particle_types:
    # Extract the dataframe for the current particle type
    particle_df = particle_dfs_without_overlaps[particle]
    
    # Filter out values based on time of flight
    filtered_df = particle_df[(particle_df[time_of_flight_column] > 0)]

    # Start a new plot for the histogram
    plt.figure(figsize=(10, 6))

    # Plot histogram for the time of flight of the current particle after filtering
    plt.hist(filtered_df[time_of_flight_column],bins=500, alpha=0.7, color='blue', edgecolor='black')

    # Labeling the plot
    plt.title(f'Time of Flight Distribution for {particle}')
    plt.xlabel('Time of Flight (ns)')
    plt.ylabel('Count')
    plt.grid(True)

    plt.show()




# Define the hit_n column name
hit_n_column = '_hit_n'

# Loop through each particle and plot its hit_n distribution
for particle, df in particle_dfs_without_overlaps.items():
    plt.figure(figsize=(10, 6))
    
    # Plot histogram for hit_n of each particle
    plt.hist(df[hit_n_column], bins=100, alpha=0.7, color='blue', edgecolor='black')
    
    # Labeling the plot
    plt.title(f'Distribution of Hit_n for {particle.capitalize()}')
    plt.xlabel('Hit_n')
    plt.ylabel('Count')
    plt.grid(True)
    
    plt.show()
    
    
    
# Start a new plot
plt.figure(figsize=(10, 6))

# Loop through each particle and plot its TOF vs Momentum distribution
for particle, df in particle_dfs_without_overlaps.items():
    plt.scatter(df[momentum_column], df[time_of_flight_column], label=particle, alpha=0.6)

# Labeling the plot
plt.title('Distributions For All Particles')
plt.xlabel('Momentum (MeV/c)')
plt.ylabel('Time of Flight (ns)')
plt.legend()
plt.grid(True)

plt.show()