# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 01:19:56 2023

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
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
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
    
    # 1. Select representative thresholds
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    high_tpr_idx = np.argmin(np.abs(tpr - 0.9))
    high_tpr_threshold = thresholds[high_tpr_idx]

    high_fpr_idx = np.argmin(np.abs(fpr - 0.9))
    high_fpr_threshold = thresholds[high_fpr_idx]

    print(f"Optimal Threshold for {particle}: {optimal_threshold}")
    print(f"High TPR Threshold for {particle}: {high_tpr_threshold}")
    print(f"High FPR Threshold for {particle}: {high_fpr_threshold}")

    # 2. Classify the test events for the optimal threshold
    TP_optimal = np.where((y_pred_prob > optimal_threshold) & (y_test == 1))
    FP_optimal = np.where((y_pred_prob > optimal_threshold) & (y_test == 0))
    TN_optimal = np.where((y_pred_prob <= optimal_threshold) & (y_test == 0))
    FN_optimal = np.where((y_pred_prob <= optimal_threshold) & (y_test == 1))

    # 3. Visualize the events for the optimal threshold
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.hist(y_pred_prob[TP_optimal], bins=30, color='g', alpha=0.6)
    plt.title(f'True Positives for {particle} (Optimal Threshold)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')

    plt.subplot(2, 2, 2)
    plt.hist(y_pred_prob[FP_optimal], bins=30, color='r', alpha=0.6)
    plt.title(f'False Positives for {particle} (Optimal Threshold)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')

    plt.subplot(2, 2, 3)
    plt.hist(y_pred_prob[TN_optimal], bins=30, color='b', alpha=0.6)
    plt.title(f'True Negatives for {particle} (Optimal Threshold)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')

    plt.subplot(2, 2, 4)
    plt.hist(y_pred_prob[FN_optimal], bins=30, color='y', alpha=0.6)
    plt.title(f'False Negatives for {particle} (Optimal Threshold)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')

    plt.tight_layout()
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
