# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 03:27:09 2023

@author: thoma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# Load the data
data = pd.read_csv('filtered_data.csv')

# Define variables for the BDT
variables = [
    "_hit_n", "_hit_avgdr", "_hit_avgpe", "_hit_firstdr", "_hit_firstpe", 
    "_hit_lastdr", "_hit_lastdz", "_hit_totpe", "_hit_lastpe",
    "_chit_frac", "_chit_gev", "_chit_width"
]

# Define the cuts for protons and pimus only
particle_cuts = {
    'proton': (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  80) & 
              (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.75) & (data['_wcn_mass'] < 1.13),
    'pimu': (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  40) & 
            (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.00) & (data['_wcn_mass'] < 0.25),
}

# Initialize list to hold data and labels
combined_data_list = []
labels_list = []

# Prepare the data for each particle type
for particle, cut in particle_cuts.items():
    # Select the data for the current particle and label it
    current_data = data[cut].copy()
    current_data['label'] = 1 if particle == 'proton' else 0
    
    # Add the data to the combined list
    combined_data_list.append(current_data)

# Concatenate the data
combined_data = pd.concat(combined_data_list)
X = combined_data[variables]
y = combined_data['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter space
param_dist = {
    "n_estimators": sp_randint(100, 600),
    "max_depth": sp_randint(3, 10),
    "min_samples_split": sp_randint(2, 11),
    "min_samples_leaf": sp_randint(1, 11),
    "learning_rate": sp_uniform(0.01, 0.3)
}

# Initialize the classifier
clf = GradientBoostingClassifier(random_state=42)

# Number of iterations for random search
n_iter_search = 20

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=5,
                                   random_state=42)

# Fit the random search model
random_search.fit(X_train, y_train)

# Print the best parameters found
print("Best parameters found during Randomized Search:")
print(random_search.best_params_)

# The best model found
best_clf = random_search.best_estimator_

# Predict probabilities
y_pred_prob = best_clf.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Proton vs. Pimu Classification')
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Print training and testing accuracies
train_acc = accuracy_score(y_train, best_clf.predict(X_train))
test_acc = accuracy_score(y_test, best_clf.predict(X_test))
print(f"Training accuracy: {train_acc:.4f}")
print(f"Testing accuracy: {test_acc:.4f}")
