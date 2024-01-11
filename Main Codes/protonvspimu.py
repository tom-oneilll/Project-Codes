# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 03:09:37 2023

@author: thoma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the data
data = pd.read_csv('filtered_data.csv')

# Define variables for the BDT
variables = [
    "_hit_n", "_hit_avgdr", "_hit_avgpe", "_hit_firstdr", "_hit_firstpe", 
     "_hit_lastdz", "_chit_gev", "_chit_width" , "_hit_lastdr", 
     "_hit_lastpe", "_hit_totpe"
]

#"_hit_lastdr" "_hit_lastpe " "_hit_totpe"

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=88)

# Normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the BDT 0.08713 180 4
clf = GradientBoostingClassifier(random_state=88,learning_rate=0.087, n_estimators=180, max_depth=10)
clf.fit(X_train, y_train)

# Predict probabilities
y_pred_prob = clf.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Proton vs. Pimu Classification')
plt.legend(loc="lower right")
plt.grid()
plt.show()

# Print training and testing accuracies
train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc = accuracy_score(y_test, clf.predict(X_test))
print(f"Training accuracy: {train_acc:.4f}")
print(f"Testing accuracy: {test_acc:.4f}")

# Plot feature importances
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 8))
plt.title("Feature importances for Proton vs. Pimu Classification")
plt.barh([variables[i] for i in indices], importances[indices], align="center")

# Annotate with numbers
for i, v in enumerate(importances[indices]):
    plt.text(v, i, f"{v:.4f}", color='black', va='center')

plt.gca().invert_yaxis()
plt.show()
