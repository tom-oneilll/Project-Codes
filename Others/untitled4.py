import pandas as pd

# Load the data
data = pd.read_csv('filtered_data.csv')

# Define particle cuts
particle_cuts = {
    'proton': (data['_wcn_mass'] > 0.75) & (data['_wcn_mass'] < 1.13),
    'pimu': (data['_wcn_mass'] > 0.00) & (data['_wcn_mass'] < 0.25),
    'kaon': (data['_wcn_mass'] > 0.39) & (data['_wcn_mass'] < 0.59),
    'ckov': data['_pass_cherenkov'] == 1
}

# Count occurrences for each particle type
particle_counts = {}
for particle, cut in particle_cuts.items():
    particle_counts[particle] = data[cut].shape[0]

# Combine all the cuts to find the "other" category
all_cuts_combined = None
for cut in particle_cuts.values():
    if all_cuts_combined is None:
        all_cuts_combined = cut
    else:
        all_cuts_combined |= cut

particle_counts["other"] = data[~all_cuts_combined].shape[0]

# Display the counts
for particle, count in particle_counts.items():
    print(f"{particle.capitalize()}: {count}")
