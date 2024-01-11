# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 04:33:44 2023

@author: thoma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, classification_report, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight

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
for i, (particle, cut) in enumerate(particle_cuts.items()):
    data.loc[cut, 'label'] = i
data = data.dropna(subset=['label'])
data['label'] = data['label'].astype(int)

# Prepare the features (X) and target (y)
X = data[variables]
y = data['label']

# Binarize the output labels for OvR
y_bin = label_binarize(y, classes=np.arange(len(particle_cuts)))

# Split into training and testing sets
X_train, X_test, y_train_bin, y_test_bin = train_test_split(X, y_bin, test_size=0.3, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Create the cost-sensitive Gradient Boosting classifier
gbt_weighted = GradientBoostingClassifier(random_state=42)

# Wrap it in OneVsRestClassifier
ovr_gbt_weighted = OneVsRestClassifier(gbt_weighted, n_jobs=-1)

# Fit the model on your data
ovr_gbt_weighted.fit(X_train_scaled, y_train_bin)

# Predict on the test set
y_score = ovr_gbt_weighted.predict_proba(X_test_scaled)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(particle_cuts)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
for i, color in zip(range(len(particle_cuts)), ['blue', 'red', 'green', 'purple']):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(list(particle_cuts.keys())[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Each Class')
plt.legend(loc="lower right")
plt.show()

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

# Predict class labels for the test set
y_pred_proba = ovr_gbt_weighted.predict_proba(X_test_scaled)

# Convert predicted probabilities to binary predictions with a threshold (e.g., 0.5)
y_pred_binary = (y_pred_proba >= 0.5).astype(int)

# Convert binary predictions back to multiclass format
y_pred = np.argmax(y_pred_binary, axis=1)

# Convert the binarized y_test back to multiclass format for comparison
y_test = np.argmax(y_test_bin, axis=1)



# Print the classification report
print(classification_report(y_test, y_pred, target_names=list(particle_cuts.keys())))
