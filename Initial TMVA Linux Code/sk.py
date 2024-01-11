#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 11:47:01 2023

@author: toneill
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import uproot

# Opening root file and converting to pandas DataFrame
file = uproot.open("trackana_p234_2004.root")
proton_tree = file["testbeamtrackana/protonTree"].pandas.df()
pimu_tree = file["testbeamtrackana/pimuTree"].pandas.df()

# Creating labels for signal and background trees
proton_tree['label'] = 1
pimu_tree['label'] = 0

# Merging both trees into a single DataFrame
data = pd.concat([proton_tree, pimu_tree])

# Variables used for training
variables = ["_hit_n", "_hit_avgdr", "_hit_avgpe", "_hit_firstdr", "_hit_firstpe", "_hit_lastdr", "_hit_lastdz", "_hit_width", "_chit_gev", "_hit_totpe", "_hit_firstdz", "_hit_lastpe", "_chit_frac"]

# Applying cuts
data = data[(data['_pass_intimehit']==1) & (data['_pass_wcntrack']==1)]

# Splitting data into features and labels
X = data[variables]
y = data['label']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating Gradient Boosting Classifier
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)

# Predicting probabilities
y_pred = clf.predict_proba(X_test)[:,1]

# Getting ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Getting AUC
roc_auc = auc(fpr, tpr)

# Plotting ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
