# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 03:01:55 2023

@author: thoma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('filtered_data.csv')

# Define variables for the BDT
variables = [
    "_hit_n", "_hit_avgdr", "_hit_avgpe", "_hit_firstdr", "_hit_firstpe", 
    "_hit_lastdr", "_hit_lastdz", "_hit_totpe", "_hit_lastpe",
    "_chit_frac", "_chit_gev", "_chit_width"
]

particle_cuts = {
    'proton': (data['_wcn_mass'] > 0.75) & (data['_wcn_mass'] < 1.13),
    'pimu': (data['_wcn_mass'] > 0.00) & (data['_wcn_mass'] < 0.25),
    'kaon': (data['_wcn_mass'] > 0.39) & (data['_wcn_mass'] < 0.59),
    'ckov': data['_pass_cherenkov'] == 1
}

for particle, cut in particle_cuts.items():
    # Prepare the data
    positive_data = data[cut]
    negative_data = data[~cut]
    
    positive_data['label'] = 1
    negative_data['label'] = 0
    
    combined_data = pd.concat([positive_data, negative_data])
    X = combined_data[variables]
    y = combined_data['label']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the BDT
    clf = GradientBoostingClassifier(learning_rate=0.08713, n_estimators=180, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    
    # Predict probabilities
    y_pred_prob = clf.predict_proba(X_test)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Determine optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Determine high TPR threshold
    high_tpr_idx = np.where(tpr > 0.9)[0][0]
    high_tpr_threshold = thresholds[high_tpr_idx]
    
    # Determine high FPR threshold
    high_fpr_idx = np.where(fpr > 0.9)[0][0]
    high_fpr_threshold = thresholds[high_fpr_idx]
    
    # Function to display samples based on a given threshold
    def display_samples(threshold, label):
        print(f"\nSamples close to {label} threshold ({threshold:.4f}) for {particle}:")
        close_samples = (y_pred_prob[:, 1] > threshold - 0.01) & (y_pred_prob[:, 1] < threshold + 0.01)
        sample_probs = y_pred_prob[close_samples]
        for i, prob in enumerate(sample_probs[:5]):  # Display first 5 samples close to the threshold
            print(f"Sample {i + 1}: Probability of 0: {prob[0]:.4f}, Probability of 1: {prob[1]:.4f}")
    
    # Display samples for different thresholds
    display_samples(optimal_threshold, 'optimal')
    display_samples(high_tpr_threshold, 'high TPR')
    display_samples(high_fpr_threshold, 'high FPR')

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {particle}')
    plt.legend(loc="lower right")
    plt.show()
