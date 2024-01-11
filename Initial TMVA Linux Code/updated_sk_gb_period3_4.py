#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:51:34 2023

@author: toneill
"""

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import uproot
import os


# PLot folder
output_folder = 'plots1'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List of root files
files = ["newprod_period3_v1.root", "newprod_period4_v1.root"]

# Define variables
variables = [
    "_hit_n", "_hit_avgdr", "_hit_avgpe", "_hit_firstdr", "_hit_firstpe",
    "_hit_lastdr", "_hit_lastdz", "_hit_totpe", "_hit_lastpe",
    "_chit_frac", "_chit_gev", "_chit_width"
]

# Initialize an empty DataFrame
all_data = pd.DataFrame()

for file_name in files:
    # Opening root file
    file = uproot.open(file_name)

    # Convert trees to pandas DataFrame
    # Loading data from the 'testbeamtrackana/variables' tree
    data = pd.DataFrame(
        file["testbeamtrackana/trackTree"].arrays(library="pd"))

    jagged_columns = [
        "_hitvec_plane", "_hitvec_nflshits", "_hitvec_x", "_hitvec_y",
        "_hitvec_xp", "_hitvec_yp", "_hitvec_z", "_hitvec_pe",
        "_hitvec_gev", "_hitvec_dr", "_hitvec_drp", "_hitvec_dx", "_hitvec_dy"
    ]

    data.drop(columns=jagged_columns, inplace=True)
    all_data = pd.concat([all_data, data])
    # Remove rows where _hit_n is 0
    all_data = all_data[all_data['_hit_n'] != 0]
    all_data = all_data[all_data['_chit_gev'] != -69]
    
all_data.to_csv('dataset.csv', index=False)


# Applying mass cuts to create separate DataFrames for each particle type

# Proton mass cuts
proton_cut = (all_data['_wcn_mass'] > 0.75) & (all_data['_wcn_mass'] < 1.13)
proton_df = all_data[proton_cut].copy()
proton_df = proton_df[variables]
proton_df.loc[:, 'label'] = 1  # label for proton_df

# Pimu mass cuts
pimu_cut = (all_data['_wcn_mass'] > 0.00) & (all_data['_wcn_mass'] < 0.25)
pimu_df = all_data[pimu_cut].copy()
pimu_df = pimu_df[variables]
pimu_df.loc[:, 'label'] = 0  # label for pimu_df

# Concatenate the two dataframes
all_data = pd.concat([proton_df, pimu_df])

# Remove rows with NaN values
all_data.dropna(inplace=True)

# Split data into features X and target y
X = all_data.drop('label', axis=1)
y = all_data['label']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define the model
clf = GradientBoostingClassifier(
    learning_rate=0.08713, n_estimators=180, max_depth=6)

# Train the model
clf.fit(X_train, y_train)

# Get predictions on the test set
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid()
plt.savefig(os.path.join(output_folder, 'roc_curve.png'))
plt.show()


# Plot feature importance
features = X.columns
importances = clf.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 10))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.grid()
plt.savefig(os.path.join(output_folder, 'feature_importance.png'))
plt.show()

# Plot histograms for each variable
bin_number = 50
variable_xlims = {
    '_hit_n': 200,
    '_hit_avgdr': 100,
    '_hit_avgpe': 1000,
    '_hit_firstdr': 140,
    '_hit_firstpe': 1000,
    '_hit_lastdr': 140,
    '_hit_lastdz': 450,
    '_hit_totpe': 50000,
    '_hit_lastpe': 2000,
    '_chit_frac': 1,
    '_chit_gev': 2,
    '_chit_width': 100}

for var in variables:
    plt.figure()

    # Determine range of the data for this variable
    min_val = min(proton_df[var].min(), pimu_df[var].min())
    max_val = max(proton_df[var].max(), pimu_df[var].max())

    # Define bins based on range
    bins = np.linspace(min_val, variable_xlims[var], bin_number)

    plt.hist(proton_df[var], bins=bins, alpha=0.5, label="proton")
    plt.hist(pimu_df[var], bins=bins, alpha=0.5, label="pimu")
    plt.legend(loc="upper right")
    plt.title(var + ' distribution')
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.grid()
    plt.tight_layout()  # this makes sure plot utilizes available space effectively
    plt.savefig(os.path.join(output_folder, f'{var}_distribution.png'))
    plt.show()


# ... [rest of your imports and existing code]


def apply_kmeans_clustering(data, n_clusters=2):
    """Apply K-means clustering on the data."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels


def visualize_clusters(data, labels, title="2D PCA of Clusters"):
    """Visualize the clusters using PCA."""
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)

    plt.figure(figsize=(10, 6))
    plt.scatter(principal_components[:, 0], principal_components[:,
                1], c=labels, cmap='viridis', s=50, alpha=0.6)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.colorbar()
    plt.grid(True)
    plt.show()


# Extracting only the pimu data
pimu_data = all_data[all_data['label'] == 0].drop('label', axis=1)

# Applying K-means clustering
labels = apply_kmeans_clustering(pimu_data)

# Visualizing the clusters using PCA
visualize_clusters(pimu_data, labels)


def recreate_histograms(proton_df, pimu_data, labels, variables):
    # Splitting pimu data into pions and muons based on K-means labels
    pion_df = pimu_data[labels == 0]
    muon_df = pimu_data[labels == 1]

    for var in variables:
        plt.figure()

        min_val = min(proton_df[var].min(),
                      pion_df[var].min(), muon_df[var].min())

        # Check if variable_xlims provides a range or a max value
        if isinstance(variable_xlims[var], list):
            max_val = variable_xlims[var][1]
        else:
            max_val = max(proton_df[var].max(),
                          pion_df[var].max(), muon_df[var].max())
            max_val = min(max_val, variable_xlims[var])

        bins = np.linspace(min_val, max_val, bin_number)

        plt.hist(proton_df[var], bins=bins, alpha=0.5, label="proton")
        plt.hist(pion_df[var], bins=bins, alpha=0.5, label="pion")
        plt.hist(muon_df[var], bins=bins, alpha=0.5, label="muon")

        plt.legend(loc="upper right")
        plt.title(var + ' distribution')
        plt.xlabel(var)
        plt.ylabel('Count')
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder,
                    f'{var}_distribution_combined.png'))
        plt.show()


recreate_histograms(proton_df, pimu_data, labels, variables)

def visualize_clusters_3d(data, labels, title="3D PCA of Clusters"):
    """Visualize the clusters using PCA in 3D."""
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(data)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], 
                         c=labels, cmap='viridis', s=50, alpha=0.6)
    
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title(title)
    fig.colorbar(scatter)
    plt.grid(True)
    plt.show()


visualize_clusters_3d(pimu_data, labels)