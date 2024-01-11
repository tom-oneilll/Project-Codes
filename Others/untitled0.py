# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 18:17:25 2023

@author: thoma
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Plot folder
output_folder = 'plots1'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load data from CSV
all_data = pd.read_csv('dataset.csv')

# Define the variables you want to save to the CSV
selected_variables = [
    "_hit_n", "_hit_avgdr", "_hit_avgpe", "_hit_firstdr", "_hit_firstpe", 
    "_hit_lastdr", "_hit_lastdz", "_hit_totpe", "_hit_lastpe",
    "_chit_frac", "_chit_gev", "_chit_width", "_wcn_mass", "_pass_cherenkov"
]

# Extract only the selected variables from the DataFrame
selected_data = all_data[selected_variables]

# Save the selected data to a CSV file
csv_path = os.path.join(output_folder, "filtered_data.csv")
selected_data.to_csv(csv_path, index=False)




# Define variables
variables = [
    "_hit_n", "_hit_avgdr", "_hit_avgpe", "_hit_firstdr", "_hit_firstpe", 
    "_hit_lastdr", "_hit_lastdz", "_hit_totpe", "_hit_lastpe",
    "_chit_frac", "_chit_gev", "_chit_width"
]

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
clf = GradientBoostingClassifier(learning_rate=0.08713, n_estimators=180, max_depth=6)

# Train the model
clf.fit(X_train, y_train)

# Get predictions on the test set
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
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

plt.figure(figsize=(10,10))
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
    '_chit_width': 100
}

for var in variables:
    plt.figure()
    min_val = min(proton_df[var].min(), pimu_df[var].min())
    max_val = variable_xlims[var] if var in variable_xlims else max(proton_df[var].max(), pimu_df[var].max())
    bins = np.linspace(min_val, max_val, bin_number)
    plt.hist(proton_df[var], bins=bins, alpha=0.5, label="proton")
    plt.hist(pimu_df[var], bins=bins, alpha=0.5, label="pimu")
    plt.legend(loc="upper right")
    plt.title(f'{var} distribution')
    plt.xlabel(var)
    plt.ylabel('Count')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'{var}_distribution.png'))
    plt.show()

def apply_kmeans_clustering(data, n_clusters=2):
    """Apply K-means clustering on the data."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels

def visualize_clusters_3d(data, labels, title="3D PCA of Clusters"):
    """Visualize the clusters using 3D PCA."""
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(data)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], c=labels, cmap='viridis', s=50, alpha=0.6)
    plt.colorbar(sc)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.set_title(title)
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, '3dcluster.png'))
    plt.show()

# Extracting only the pimu data
pimu_data = all_data[all_data['label'] == 0].drop('label', axis=1)

# Applying K-means clustering
labels = apply_kmeans_clustering(pimu_data)

# Visualizing the clusters using 3D PCA
visualize_clusters_3d(pimu_data, labels)

def recreate_histograms(proton_df, pimu_data, labels, variables):
    # Splitting pimu data into pions and muons based on K-means labels
    pion_df = pimu_data[labels == 0]
    muon_df = pimu_data[labels == 1]
    
    for var in variables:
        plt.figure()
        min_val = min(proton_df[var].min(), pion_df[var].min(), muon_df[var].min())
        max_val = variable_xlims[var] if var in variable_xlims else max(proton_df[var].max(), pion_df[var].max(), muon_df[var].max())
        bins = np.linspace(min_val, max_val, bin_number)
        plt.hist(proton_df[var], bins=bins, alpha=0.5, label="proton")
        plt.hist(pion_df[var], bins=bins, alpha=0.5, label="pion")
        plt.hist(muon_df[var], bins=bins, alpha=0.5, label="muon")
        
        plt.legend(loc="upper right")
        plt.title(f'{var} distribution')
        plt.xlabel(var)
        plt.ylabel('Count')
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{var}_distribution_combined.png'))
        plt.show()

recreate_histograms(proton_df, pimu_data, labels, variables)

