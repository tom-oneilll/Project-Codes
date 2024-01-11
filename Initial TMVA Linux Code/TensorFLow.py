#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 02:02:45 2023

@author: toneill
"""

import uproot
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load ROOT file
file = uproot.open("trackana_p234_2004.root")

# Load trees into pandas dataframes
proton_df = file["testbeamtrackana/protonTree"].pandas.df()
pimu_df = file["testbeamtrackana/pimuTree"].pandas.df()

# Merge and create labels
proton_df['label'] = 1
pimu_df['label'] = 0
df = pd.concat([proton_df, pimu_df])

# Apply cuts
df = df[(df['_pass_intimehit'] == 1) & (df['_pass_wcntrack'] == 1)]

# Define features
features = ["_hit_n", "_hit_avgdr", "_hit_avgpe", "_hit_firstdr", "_hit_firstpe",
            "_hit_lastdr", "_hit_lastdz", "_hit_width", "_chit_gev", "_hit_totpe",
            "_hit_firstdz", "_hit_lastpe", "_chit_frac"]

# Prepare data
X = df[features]
y = df['label']

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit model to the training data
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Predict probabilities for ROC curve
y_pred_proba = model.predict(X_test).ravel()

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
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
