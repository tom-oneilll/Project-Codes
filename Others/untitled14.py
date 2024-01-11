# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:36:13 2023

@author: thoma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import roc_curve, auc, classification_report, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, label_binarize
from imblearn.over_sampling import SMOTE

# Load the data
data = pd.read_csv('filtered_data.csv')

# Define variables for the models
variables = [
    "_hit_n", "_hit_avgdr", "_hit_avgpe", "_hit_firstdr", "_hit_firstpe", 
    "_hit_lastdr", "_hit_lastdz", "_hit_totpe", "_hit_lastpe",
     "_chit_gev", "_chit_width", "_hit_width"
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

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train_bin)

# One-vs-One Classifier
base_classifier = GradientBoostingClassifier(random_state=42)
ovo_classifier = OneVsOneClassifier(base_classifier)
ovo_classifier.fit(X_train_smote, np.argmax(y_train_smote, axis=1))

# Predict on the test set
y_pred = ovo_classifier.predict(X_test_scaled)
y_test = np.argmax(y_test_bin, axis=1)

# Classification report
print(classification_report(y_test, y_pred, target_names=list(particle_cuts.keys())))

# ROC Curves
y_pred_proba = ovo_classifier.decision_function(X_test_scaled)
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = len(particle_cuts)
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'purple']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of {0} (area = {1:0.2f})'.format(list(particle_cuts.keys())[i], roc_auc[i]))

# Plot Precision-Recall curve for each class and compute Average Precision (AP)
plt.figure(figsize=(10, 8))
for i, color in zip(range(len(particle_cuts)), ['blue', 'red', 'green', 'purple']):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    ap_score = average_precision_score(y_test_bin[:, i], y_score[:, i])
    plt.plot(recall, precision, color=color, lw=2,
             label='Class {0} AP: {1:0.2f}'.format(list(particle_cuts.keys())[i], ap_score))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')
plt.legend(loc="best")
plt.show()

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('One-vs-One ROC Curves')
plt.legend(loc="lower right")
plt.show()
