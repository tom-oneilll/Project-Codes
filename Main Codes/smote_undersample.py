# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 04:13:02 2023

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
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# Load the data
data = pd.read_csv('filtered_data.csv')

# Define variables for the models
variables = [
    "_hit_n", "_hit_avgdr", "_hit_avgpe", "_hit_firstdr", "_hit_firstpe", 
    "_hit_lastdr", "_hit_lastdz", "_hit_totpe", "_hit_lastpe",
    "_chit_frac", "_chit_gev", "_chit_width","_hit_width"
]

particle_cuts = {
    'proton':     (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  80) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.75) & (data['_wcn_mass'] < 1.13),
    'pimu':       (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  40) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.00) & (data['_wcn_mass'] < 0.25), 
    'kaon':       (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  70) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.39) & (data['_wcn_mass'] < 0.59),
    'electron':   (data['_pass_cherenkov'] == 1) & (data['_tof_time'] >  32) & (data['_tof_time'] <  35) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000)
}

# Initialize the label column
data['label'] = np.nan

# Assign labels based on the cuts
for particle, cut in particle_cuts.items():
    data.loc[cut, 'label'] = list(particle_cuts.keys()).index(particle)

# Remove events that do not pass any cut
data = data.dropna(subset=['label'])

# Convert labels to integer type
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

# Identify the Kaon class
kaon_index = list(particle_cuts.keys()).index('kaon')


# Checking class counts before SMOTE
print("Class counts before SMOTE:", np.sum(y_train_bin, axis=0))

# Identify the indices for Kaon and Electron classes
kaon_index = list(particle_cuts.keys()).index('kaon')
electron_index = list(particle_cuts.keys()).index('electron')
proton_index = list(particle_cuts.keys()).index('proton')
pimu_index = list(particle_cuts.keys()).index('pimu')

# Define custom sampling strategy for SMOTE
sampling_strategy_smote = {
    kaon_index: int(y_train_bin[:, kaon_index].sum() * 1),
    electron_index: int(y_train_bin[:, electron_index].sum() * 1 )
}


# Define undersampling strategy for RandomUnderSampler
sampling_strategy_under = {
    proton_index: int(y_train_bin[:, proton_index].sum() * 0.75),  # Assuming 0 is the index for Proton
    pimu_index: int(y_train_bin[:,pimu_index].sum() * 0.75)   # Assuming 1 is the index for Pimu
}


# Create a pipeline that first applies SMOTE and then applies RandomUnderSampler
resample_pipeline = Pipeline([
    ('SMOTE', SMOTE(sampling_strategy=sampling_strategy_smote, random_state=42)),
    ('RandomUnderSampler', RandomUnderSampler(sampling_strategy=sampling_strategy_under, random_state=42))
])
# Apply the resampling pipeline to the training data
X_train_resampled, y_train_resampled = resample_pipeline.fit_resample(X_train_scaled, y_train_bin)

# Checking class counts after applying SMOTE and RandomUnderSampler
print("Class counts after resampling:", np.sum(y_train_resampled, axis=0))

# Create the OvR strategy classifier
clf = OneVsRestClassifier(GradientBoostingClassifier(random_state=42))
clf.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_score = clf.predict_proba(X_test_scaled)

# Compute and plot ROC curve and ROC area for each class
plt.figure()
for i, color in zip(range(len(particle_cuts)), ['blue', 'red', 'green', 'purple']):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(list(particle_cuts.keys())[i], roc_auc))

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
y_pred = clf.predict(X_test_scaled)
y_test = np.argmax(y_test_bin, axis=1)
y_pred = np.argmax(y_pred, axis=1)

# Print the classification report
print(classification_report(y_test, y_pred, target_names=list(particle_cuts.keys())))