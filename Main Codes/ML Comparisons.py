# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 06:53:28 2023

@author: thoma
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

classifiers = {
    'Gradient Boosted Decision Trees': GradientBoostingClassifier(learning_rate=0.08713, n_estimators=180, max_depth=6, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Neural Network': MLPClassifier(random_state=42)
}


for particle, cut in particle_cuts.items():
    print(f"Evaluating for {particle}...")
    
    # Prepare the data
    positive_data = data[cut].copy()
    negative_data = data[~cut].copy()
    
    positive_data['label'] = 1
    negative_data['label'] = 0
    
    combined_data = pd.concat([positive_data, negative_data])
    X = combined_data[variables]
    y = combined_data['label']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    for name, clf in classifiers.items():
        print(f"Training {name} for {particle}...")
        clf.fit(X_train, y_train)
        
        # Predict probabilities
        y_pred_prob = clf.predict_proba(X_test)[:, 1]
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {particle} using {name}')
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig(f"ROC_{particle}_{name.replace(' ', '_')}.png")  # Save the ROC curve to a file
        plt.close()
        
        # If the classifier has feature importances, plot them
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 8))
            plt.title(f"Feature importances for {particle} using {name}")
            plt.barh(variables, importances[indices], align="center")
            
            # Annotate with numbers
            for i, v in enumerate(importances[indices]):
                plt.text(v, i, f"{v:.4f}", va='center')
                
            plt.gca().invert_yaxis()
            plt.savefig(f"FeatureImportance_{particle}_{name.replace(' ', '_')}.png")  # Save the feature importance plot to a file
            plt.close()
        
        # Print training and testing accuracies
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"Training accuracy for {name}: {train_acc:.4f}")
        print(f"Testing accuracy for {name}: {test_acc:.4f}")
        print("------------------------------------------------")
