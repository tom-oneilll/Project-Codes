#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 03:13:44 2023

@author: toneill
"""

import uproot
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load ROOT file
file = uproot.open("trackana_p234_2004.root")

# Load trees into pandas dataframes
proton_tree = file["testbeamtrackana/protonTree"]
proton_df = pd.DataFrame(proton_tree.arrays())

pimu_tree = file["testbeamtrackana/pimuTree"]
pimu_df = pd.DataFrame(pimu_tree.arrays())

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
X = df[features].values
y = df['label'].values

# Standardize features manually
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Split data into training and test sets manually
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
split_idx = int(0.8 * X.shape[0])
train_idx, test_idx = indices[:split_idx], indices[split_idx:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Convert data to tensors
X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)
X_test = torch.tensor(X_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.float)

# Define model architecture
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

# Define loss and optimizer
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
for epoch in range(10):
    y_pred = model(X_train)
    loss = loss_fn(y_pred.squeeze(), y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Predict probabilities for ROC curve
y_pred_proba = model(X_test).detach().numpy()

# Compute ROC curve and ROC area manually
fpr = []
tpr = []
thresholds = np.linspace(0, 1, 100)
for threshold in thresholds:
    tp = np.sum((y_pred_proba >= threshold) & (y_test == 1))
    fp = np.sum((y_pred_proba >= threshold) & (y_test == 0))
    fn = np.sum((y_pred_proba < threshold) & (y_test == 1))
    tn = np.sum((y_pred_proba < threshold) & (y_test == 0))
    fpr.append(fp / (fp + tn))
    tpr.append(tp / (tp + fn))
roc_auc = np.trapz(tpr, fpr)

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
