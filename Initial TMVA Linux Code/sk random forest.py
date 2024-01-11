#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 03:26:49 2023

@author: toneill
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import uproot

# Opening root file
file = uproot.open("trackana_p234_2004.root")

# Define variables
variables = ["_hit_n", "_hit_avgdr", "_hit_avgpe", "_hit_firstdr", "_hit_firstpe", "_hit_lastdr", "_hit_lastdz", "_hit_width", "_chit_gev", "_hit_totpe", "_hit_firstdz", "_hit_lastpe", "_chit_frac"]

# Convert trees to pandas DataFrame
proton_tree = pd.DataFrame(file["testbeamtrackana/protonTree"].arrays(library="pd"))
proton_tree = proton_tree[variables]
proton_tree['label'] = 1  # label for proton_tree

pimu_tree = pd.DataFrame(file["testbeamtrackana/pimuTree"].arrays(library="pd"))
pimu_tree = pimu_tree[variables]
pimu_tree['label'] = 0  # label for pimu_tree

# Concatenate the two dataframes
data = pd.concat([proton_tree, pimu_tree])

# Remove rows with NaN values
data.dropna(inplace=True)

# Split data into features X and target y
X = data.drop('label', axis=1)
y = data['label']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model, use RandomForestClassifier instead of GradientBoostingClassifier
clf = RandomForestClassifier(n_estimators=180, max_depth=5, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Get predictions on the test set
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid()
plt.show()
