# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 04:32:51 2023

@author: thoma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
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

particle_cuts = {
    'proton':     (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  80) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.75) & (data['_wcn_mass'] < 1.13),
    'pimu':       (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  40) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.00) & (data['_wcn_mass'] < 0.25), 
    'kaon':       (data['_pass_cherenkov'] == 0) & (data['_tof_time'] >  32) & (data['_tof_time'] <  70) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000) & (data['_wcn_mass'] > 0.39) & (data['_wcn_mass'] < 0.59),
    'electron':   (data['_pass_cherenkov'] == 1) & (data['_tof_time'] >  32) & (data['_tof_time'] <  35) & (data['_wcn_p'] > 200) & (data['_wcn_p'] < 2000)
}

# Initialize the label column
data['label'] = np.nan

# Assign labels based on the cuts
for i, (particle, cut) in enumerate(particle_cuts.items()):
    data.loc[cut, 'label'] = i

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

# Apply SMOTE specifically for the Kaon class
smote = SMOTE(random_state=42)
kaon_index = list(particle_cuts.keys()).index('kaon')
X_train_smote, y_train_kaon_smote = smote.fit_resample(X_train_scaled, y_train_bin[:, kaon_index])

# Process other classes in a loop if needed
# [Your loop for processing other classes]

# Combine the SMOTE-enhanced Kaon data with the rest of the training data
# Note: This step depends on how you've processed other classes
# [Code to combine SMOTE-enhanced Kaon data with other class data]

# Create the OvR strategy classifier
clf = OneVsRestClassifier(GradientBoostingClassifier(random_state=42))
clf.fit(X_train_combined, y_train_combined)

# Predict on the test set
y_score = clf.predict_proba(X_test_scaled)

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
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curve for the micro-average
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Micro-average ROC curve')
plt.legend(loc="lower right")
plt.show()

# Predict class labels for the test set
y_pred = clf.predict(X_test_scaled)

y_test = np.argmax(y_test_bin, axis=1)


y_pred = np.argmax(y_pred, axis=1)

#Print the classification report
print(classification_report(y_test, y_pred, target_names=list(particle_cuts.keys())))


"""

# Find the optimal threshold for each class
optimal_thresholds = np.zeros(len(particle_cuts))
for i in range(len(particle_cuts)):
    precision, recall, thresholds = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    epsilon = 1e-8
    f1_scores = 2 * recall * precision / (recall + precision + epsilon)
    # Find the threshold that maximizes the F1-score
    max_index = np.nanargmax(f1_scores)  # Avoid NaN values with nanargmax
    optimal_thresholds[i] = thresholds[max_index]

# Adjust prediction based on the new optimal thresholds
adjusted_pred = np.array([[1 if y_score[j, i] >= optimal_thresholds[i] else 0 for i in range(len(particle_cuts))] for j in range(len(y_test))])

# Now you can generate the classification report with the adjusted predictions
print(classification_report(y_test, np.argmax(adjusted_pred, axis=1), target_names=list(particle_cuts.keys())))

"""



