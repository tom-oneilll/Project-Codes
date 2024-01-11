# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 07:21:53 2023

@author: thoma
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import itertools
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

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
    #'kaon':       (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  70) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.39) & (data['_wcn_mass'] < 0.59),
    'electron':   (data['_pass_cherenkov'] == 1) & (data['_tof_time'] >  32) & (data['_tof_time'] <  35) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000)
}

# Define weights for each class
class_weights = {'proton': 1, 'pimu': 1, 'electron': 1}  # Adjust these weights as needed


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

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Function to train binary classifiers for each particle pair
def train_binary_classifiers(X_train, y_train, particle_cuts):
    binary_classifiers = {}
    for (particle1, particle2) in itertools.combinations(particle_cuts.keys(), 2):
        pair_label = f"{particle1}_vs_{particle2}"
        print(f"Training for {pair_label}")

        # Prepare binary labels for current pair
        binary_labels = np.where((y_train == list(particle_cuts.keys()).index(particle1)) | 
                                 (y_train == list(particle_cuts.keys()).index(particle2)), 
                                 y_train, np.nan)
        valid_indices = ~np.isnan(binary_labels)
        binary_labels = np.where(binary_labels[valid_indices] == list(particle_cuts.keys()).index(particle1), 1, 0)

        # Train GBDT
        gbdt_classifier = GradientBoostingClassifier(random_state=42)
        binary_classifiers[pair_label] = clone(gbdt_classifier).fit(X_train[valid_indices], binary_labels)
    
    return binary_classifiers

# Function for combined OvO prediction
def ovo_predict_weighted(X, binary_classifiers, class_labels, class_weights):
    votes = np.zeros((X.shape[0], len(class_labels)))

    # For each classifier, make predictions and accumulate weighted votes
    for pair_label, classifier in binary_classifiers.items():
        particle1, particle2 = pair_label.split('_vs_')
        indices_1, indices_2 = class_labels.index(particle1), class_labels.index(particle2)

        predictions = classifier.predict(X)
        for i, pred in enumerate(predictions):
            if pred == 1:  # Weighted vote for particle1
                votes[i, indices_1] += class_weights[particle1]
            else:  # Weighted vote for particle2
                votes[i, indices_2] += class_weights[particle2]

    # Determine final prediction based on weighted votes
    final_predictions = np.argmax(votes, axis=1)
    return final_predictions


# Train binary classifiers
binary_classifiers = train_binary_classifiers(X_train, y_train, particle_cuts)

# Class labels in the same order as used for training
class_labels = list(particle_cuts.keys())

# Use the weighted OvO classifier for predictions on the test set
ovo_predictions_weighted = ovo_predict_weighted(X_test, binary_classifiers, class_labels, class_weights)

# Evaluate performance
print("Confusion Matrix:")
print(confusion_matrix(y_test, ovo_predictions_weighted))

print("\nClassification Report:")
print(classification_report(y_test, ovo_predictions_weighted, target_names=class_labels))

print("\nAccuracy Score:")
print(accuracy_score(y_test, ovo_predictions_weighted))
