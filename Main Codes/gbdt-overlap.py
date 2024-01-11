# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 05:10:31 2023

@author: thoma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
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

# Define overlap cuts
overlap_cuts = {
    'proton_ckov': (particle_cuts['proton'] & particle_cuts['ckov']),
    'pimu_ckov': (particle_cuts['pimu'] & particle_cuts['ckov']),
    'kaon_ckov': (particle_cuts['kaon'] & particle_cuts['ckov'])
}

# Remove overlaps from each particle group
for particle, cut in particle_cuts.items():
    for overlap, overlap_cut in overlap_cuts.items():
        if particle in overlap:
            data.loc[overlap_cut, particle] = False

for particle, cut in particle_cuts.items():
    # Prepare the data
    positive_data = data[cut].copy()
    negative_data = data[~cut].copy()
    
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
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {particle}')
    plt.legend(loc="lower right")
    plt.show()
    
    # Print training and testing accuracies
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Training accuracy for {particle}: {train_acc:.4f}")
    print(f"Testing accuracy for {particle}: {test_acc:.4f}")
    
    # Plot feature importances
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 8))
    plt.title(f"Feature importances for {particle}")
    plt.barh(variables, importances[indices], align="center")
    
    # Annotate with numbers
    for i, v in enumerate(importances[indices]):
        plt.text(v, i, f"{v:.4f}", va='center')
        
    plt.gca().invert_yaxis()
    plt.show()
    
    
# List of particle pairs for comparison
particle_pairs = [('proton', 'pimu'), ('proton', 'kaon'), ('proton', 'ckov'),
                  ('pimu', 'kaon'), ('pimu', 'ckov'), ('kaon', 'ckov')]

for particle1, particle2 in particle_pairs:
    # Prepare the data
    positive_data = data[particle_cuts[particle1]].copy()
    negative_data = data[particle_cuts[particle2]].copy()
    
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
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {particle1} vs {particle2}')
    plt.legend(loc="lower right")
    plt.show()
    
    # Print training and testing accuracies
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Training accuracy for {particle1} vs {particle2}: {train_acc:.4f}")
    print(f"Testing accuracy for {particle1} vs {particle2}: {test_acc:.4f}")
    
    # Plot feature importances
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 8))
    plt.title(f"Feature importances for {particle1} vs {particle2}")
    plt.barh(variables, importances[indices], align="center")
    
    # Annotate with numbers
    for i, v in enumerate(importances[indices]):
        plt.text(v, i, f"{v:.4f}", va='center')
        
    plt.gca().invert_yaxis()
    plt.show()
