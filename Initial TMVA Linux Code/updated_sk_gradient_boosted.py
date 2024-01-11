#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 02:38:40 2023

@author: toneill
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import uproot

# Opening root file
file = uproot.open("newprod_period4_v1.root")

# Define variables
variables = ["_hit_n", "_hit_avgdr", "_hit_avgpe", "_hit_firstdr", "_hit_firstpe", "_hit_lastdr", "_hit_lastdz", "_hit_totpe", "_hit_firstdz", "_hit_lastpe", "_chit_frac","_chit_gev","_chit_width"]

# Convert trees to pandas DataFrame

# Loading data from the 'testbeamtrackana/variables' tree
data = pd.DataFrame(file["testbeamtrackana/trackTree"].arrays(library="pd"))

jagged_columns = [
    "_hitvec_plane", "_hitvec_nflshits", "_hitvec_x", "_hitvec_y", 
    "_hitvec_xp", "_hitvec_yp", "_hitvec_z", "_hitvec_pe", 
    "_hitvec_gev", "_hitvec_dr", "_hitvec_drp", "_hitvec_dx", "_hitvec_dy"
]

data.drop(columns=jagged_columns, inplace=True)
"""
def is_jagged(series):
    #Check if a pandas Series contains jagged arrays.
    # Get lengths of the arrays in the series
    lengths = series.apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else None)
    
    # Check if there's more than one unique length
    unique_lengths = lengths.dropna().unique()
    return len(unique_lengths) > 1

for col in data.columns:
    if is_jagged(data[col]):
        print(f"Column {col} has jagged arrays.")
"""
# Applying mass cuts to create separate DataFrames for each particle type

# Proton mass cuts
proton_cut = (data['_wcn_mass'] > 0.75) & (data['_wcn_mass'] < 1.13)
proton_df = data[proton_cut].copy()
proton_df = proton_df[variables]
proton_df.loc[:, 'label'] = 1 # label for proton_df

# Pimu mass cuts
pimu_cut = (data['_wcn_mass'] > 0.07) & (data['_wcn_mass'] < 0.21)
pimu_df = data[pimu_cut].copy()
pimu_df = pimu_df[variables]
pimu_df.loc[:, 'label'] = 0 # label for pimu_df

# Kaon mass cuts
"""kaon_cut = (data['_wcn_mass'] > 0.39) & (data['_wcn_mass'] < 0.59)
kaon_df = data[kaon_cut]


# Convert the column values to binary (0 or 1)
data['_pass_cherenkov_binary'] = (data['_pass_cherenkov'] > 1).astype(int)

# Now, filter the data based on the binary column
ckov_df = data[data['_pass_cherenkov_binary'] == 1]
"""

# Concatenate the two dataframes
data = pd.concat([proton_df, pimu_df])

# Remove rows with NaN values
data.dropna(inplace=True)



# Split data into features X and target y
X = data.drop('label', axis=1)
y = data['label']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
clf = GradientBoostingClassifier()



# Train the model
clf.fit(X_train, y_train)

# Get predictions on the test set
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid()
plt.show()
