# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 07:09:18 2023

@author: thoma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, classification_report, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import itertools

# Load the data
data = pd.read_csv('filtered_data.csv')

# Define variables for the models
variables = [
    "_hit_n", "_hit_avgdr", "_hit_avgpe", "_hit_firstdr", "_hit_firstpe", 
    "_hit_lastdr", "_hit_lastdz", "_hit_totpe", "_hit_lastpe",
    "_chit_gev", "_chit_width", "_hit_width"
]

# Define particle cuts
particle_cuts = {
    'proton':     (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  80) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.75) & (data['_wcn_mass'] < 1.13),
    'pimu':       (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  40) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.00) & (data['_wcn_mass'] < 0.25), 
    'kaon':       (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  70) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.39) & (data['_wcn_mass'] < 0.59),
    'electron':   (data['_pass_cherenkov'] == 1) & (data['_tof_time'] >  32) & (data['_tof_time'] <  35) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000)
}

# Initialize and assign labels
data['label'] = np.nan
for particle, cut in particle_cuts.items():
    data.loc[cut, 'label'] = list(particle_cuts.keys()).index(particle)
data = data.dropna(subset=['label'])
data['label'] = data['label'].astype(int)

# Prepare features
X = data[variables]
y = data['label']

# Normalize the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Function to plot ROC and Precision-Recall Curves
def plot_curves(y_test, y_scores, title):
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    average_precision = average_precision_score(y_test, y_scores)


    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    plt.figure()
    plt.plot(recall, precision, color='blue', alpha=0.8, label='AP = %0.2f' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve')
    plt.legend(loc="lower left")

    plt.suptitle(title)
    plt.show()

# Iterate over each pair of particles
for (particle1, particle2) in itertools.combinations(particle_cuts.keys(), 2):
    print(f"Training for {particle1} vs {particle2}")

    # Prepare binary labels for current pair
    binary_labels = np.where((y == list(particle_cuts.keys()).index(particle1)) | (y == list(particle_cuts.keys()).index(particle2)), y, np.nan)
    valid_indices = ~np.isnan(binary_labels)
    binary_labels = np.where(binary_labels[valid_indices] == list(particle_cuts.keys()).index(particle1), 1, 0)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled[valid_indices], binary_labels, test_size=0.3, random_state=42)

    # Train GBDT
    gbdt_classifier = GradientBoostingClassifier(random_state=42)
    gbdt_classifier.fit(X_train, y_train)

    # Predict scores
    y_scores = gbdt_classifier.decision_function(X_test)

    # Plot ROC and Precision-Recall Curves
    plot_curves(y_test, y_scores, f"{particle1} vs {particle2}")
