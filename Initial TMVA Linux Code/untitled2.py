#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 12:07:11 2023

@author: toneill
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
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

# Handle NaN values
data.fillna(data.mean(), inplace=True)

# Split data into features X and target y
X = data.drop('label', axis=1)
y = data['label']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
clf = GradientBoostingClassifier()

# Train the model
clf.fit(X_train, y_train)

# Get predictions on the test set
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

# Set up the parameters
param_dist = {"learning_rate": sp_randFloat(),
              "n_estimators": sp_randInt(100, 200),
              "max_depth": sp_randInt(3, 5)}

from sklearn.model_selection import ParameterSampler, cross_val_score
from tqdm.auto import tqdm

# Assuming that param_dist is your parameter grid and clf is your classifier
param_list = list(ParameterSampler(param_dist, n_iter=10, random_state=0))

best_score = -np.inf
best_params = None

for params in tqdm(param_list, total=len(param_list), desc="Random Search Progress:"):
    clf.set_params(**params)
    score = cross_val_score(clf, X_train, y_train, cv=5).mean()
    if score > best_score:
        best_score = score
        best_params = params

clf.set_params(**best_params)
clf.fit(X_train, y_train)

