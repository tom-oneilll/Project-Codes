# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 05:09:31 2023

@author: thoma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, label_binarize
from imblearn.over_sampling import SMOTE

# Load the data
data = pd.read_csv('filtered_data.csv')

# Define variables for the models
variables = [
    "_hit_n", "_hit_avgdr", "_hit_avgpe", "_hit_firstdr", "_hit_firstpe", 
    "_hit_lastdr", "_hit_lastdz", "_hit_totpe", "_hit_lastpe",
    "_chit_frac", "_chit_gev", "_chit_width","_hit_width"
]

# Define particle cuts
particle_cuts = {
    'proton':     (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  80) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.75) & (data['_wcn_mass'] < 1.13),
    'pimu':       (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  40) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.00) & (data['_wcn_mass'] < 0.25), 
    'kaon':       (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  70) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.39) & (data['_wcn_mass'] < 0.59),
    'electron':   (data['_pass_cherenkov'] == 1) & (data['_tof_time'] >  32) & (data['_tof_time'] <  35) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000)
}

# Initialize and assign labels
data['label'] = np.nan
for particle, cut in particle_cuts.items():
    data.loc[cut, 'label'] = list(particle_cuts.keys()).index(particle)
data = data.dropna(subset=['label'])
data['label'] = data['label'].astype(int)

# Prepare features and target
X = data[variables]
y = data['label']
y_bin = label_binarize(y, classes=np.arange(len(particle_cuts)))

# Split into training and testing sets
X_train, X_test, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.3, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE
kaon_index = list(particle_cuts.keys()).index('kaon')
electron_index = list(particle_cuts.keys()).index('electron')
sampling_strategy = {
    kaon_index: int(y_train_bin[:, kaon_index].sum() * 25),
    electron_index: int(y_train_bin[:, electron_index].sum() * 10)
}
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train_bin)

# GridSearchCV for hyperparameter tuning
param_grid = {
    'estimator__n_estimators': [100, 200, 300],
    'estimator__max_depth': [3, 5, 7],
    'estimator__learning_rate': [0.01, 0.1, 0.2]
}
gbt = GradientBoostingClassifier(random_state=42)
clf = OneVsRestClassifier(gbt)
grid_search = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1, scoring='f1_macro')
grid_search.fit(X_train_smote, y_train_smote)

# Best model
best_clf = grid_search.best_estimator_

# Predict on the test set and compute metrics
y_pred = best_clf.predict(X_test_scaled)
y_test = np.argmax(y_test_bin, axis=1)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred, target_names=list(particle_cuts.keys())))

# Plotting ROC curves
plt.figure()
for i, color in zip(range(len(particle_cuts)), ['blue', 'red', 'green', 'purple']):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(list(particle_cuts.keys())[i], roc_auc))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Each Class')
plt.legend(loc="lower right")
plt.show()
