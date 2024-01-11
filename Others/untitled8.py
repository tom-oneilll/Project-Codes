# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 00:50:50 2023

@author: thoma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('dataset.csv')

# Define variables for the models
variables = [
    "_hit_n", "_hit_avgdr", "_hit_avgpe", "_hit_firstdr", "_hit_firstpe", 
    "_hit_lastdr", "_hit_lastdz", "_hit_totpe", "_hit_lastpe",
    "_chit_frac", "_chit_gev", "_chit_width","_hit_width"
]

particle_cuts = {
    'proton':     (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  80) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.75) & (data['_wcn_mass'] < 1.13),
    'pimu':       (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  40) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.00) & (data['_wcn_mass'] < 0.25), 
    'kaon':       (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  70) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.39) & (data['_wcn_mass'] < 0.59),
    'electron':   (data['_pass_cherenkov'] == 1) & (data['_tof_time'] >  32) & (data['_tof_time'] <  35) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000)
}

# Define particle cuts based on conditions provided in the particle_cuts dictionary
proton_cut = particle_cuts['proton']
pimu_cut = particle_cuts['pimu']
kaon_cut = particle_cuts['kaon']
electron_cut = particle_cuts['electron']

# Define classifiers
classifiers = {
    'Gradient Boosted Decision Trees': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    'Random Forest': RandomForestClassifier(),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64, 64), activation='relu', random_state=42)
}

# Function to evaluate classifiers
def evaluate_classifiers(X, y, classifiers, scenario):
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train and evaluate classifiers
    for name, clf in classifiers.items():
        print(f"Scenario: {scenario} - Training {name}...")
        
        start_time = time.time()
        clf.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Predict probabilities and labels
        y_pred_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        # Compute ROC curve and ROC area
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {scenario} - {name}')
        plt.legend(loc="lower right")
        plt.show()

        # Compute accuracy and F1 score
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)

        # Print results
        print(f"Testing accuracy for {name}: {test_acc:.4f}")
        print(f"Testing F1 Score for {name}: {test_f1:.4f}")
        print(f"Time taken to train {name}: {training_time:.4f} seconds")
        print(f"AUC for {name}: {roc_auc:.4f}")
        print("------------------------------------------------")

# Create a binary label for the 'proton' vs 'pimu' scenario
data['label_pimu'] = 0  # Default label
data.loc[pimu_cut, 'label_pimu'] = 1  # Label for pimu

# Extract the positive and negative data
positive_data_pimu = data[proton_cut].copy()
negative_data_pimu = data[pimu_cut].copy()

# Combine, shuffle, and create the dataset for 'proton' vs 'pimu'
combined_data_pimu = pd.concat([positive_data_pimu, negative_data_pimu]).sample(frac=1, random_state=42)
X_pimu = combined_data_pimu[variables]
y_pimu = combined_data_pimu['label_pimu']

# Evaluate classifiers for 'proton' vs 'pimu'
evaluate_classifiers(X_pimu, y_pimu, classifiers, "Proton vs Pimu")

# Create a combined cut for all identified particles
identified_cut = pimu_cut | kaon_cut | electron_cut

# Create a binary label for the 'proton' vs 'all identified particles' scenario
data['label_all'] = 0  # Default label
data.loc[proton_cut, 'label_all'] = 1  # Label for proton

# Extract the positive and negative data for 'proton' vs 'all identified particles'
negative_data_identified = data[identified_cut & ~proton_cut].copy()

# Combine, shuffle, and create the dataset for 'proton' vs 'all identified particles'
combined_data_all = pd.concat([data[proton_cut], negative_data_identified]).sample(frac=1, random_state=42)
X_all = combined_data_all[variables]
y_all = combined_data_all['label_all']

# Evaluate classifiers for 'proton' vs 'all identified particles'
evaluate_classifiers(X_all, y_all, classifiers, "Proton vs All Identified Particles")
