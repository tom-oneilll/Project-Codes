# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 23:39:48 2023

@author: thoma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score,  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('filtered_data.csv')

# Define variables for the models
variables = [
    "_hit_n", "_hit_avgdr", "_hit_avgpe", "_hit_firstdr", "_hit_firstpe", 
    "_hit_lastdr", "_hit_lastdz", "_hit_totpe", "_hit_lastpe",
    "_chit_frac", "_chit_gev", "_chit_width", "_hit_width"
]

# Define particle cuts for proton and pimu
particle_cuts = {
    'proton': (data['_pass_cherenkov'] == 0) & (data['_tof_time'] > 32) & (data['_tof_time'] < 80) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.75) & (data['_wcn_mass'] < 1.13),
    'pimu': (data['_pass_cherenkov'] == 0) & (data['_tof_time'] > 32) & (data['_tof_time'] < 40) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.00) & (data['_wcn_mass'] < 0.25)
}

classifiers = {
    'Gradient Boosted Decision Trees': GradientBoostingClassifier(random_state=42),
    'Random Forest': RandomForestClassifier( random_state=42),
    'Neural Network': MLPClassifier(random_state=42)
}

# Prepare the data for 'proton' and 'pimu'
positive_data = data[particle_cuts['proton']].copy()
negative_data = data[particle_cuts['pimu']].copy()

positive_data['label'] = 1
negative_data['label'] = 0

combined_data = pd.concat([positive_data, negative_data])
X = combined_data[variables]
y = combined_data['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate classifiers
for name, clf in classifiers.items():
    print(f"Training {name}...")
    
    start_time = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Predict probabilities
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for proton vs pimu using {name}')
    plt.legend(loc="lower right")
    plt.show()
    
    # Print training and testing accuracies and F1 score
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    train_f1 = f1_score(y_train, clf.predict(X_train))
    test_f1 = f1_score(y_test, y_pred)
    
    print(f"Training accuracy for {name}: {train_acc:.4f}")
    print(f"Testing accuracy for {name}: {test_acc:.4f}")
    print(f"Training F1 Score for {name}: {train_f1:.4f}")
    print(f"Testing F1 Score for {name}: {test_f1:.4f}")
    print(f"Time taken to train {name}: {training_time:.4f} seconds")
    print("------------------------------------------------")
