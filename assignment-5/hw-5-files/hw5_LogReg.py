#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:02:35 2022

@author: milos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	roc_auc_score,
	roc_curve,
	precision_recall_curve
)

# load data sets

print('Loading training set')

df_train = pd.read_csv('pima_train.csv')
print(df_train.head())

X_train = df_train.values[:,0:8]
Y_train = df_train.values[:,8]
print(X_train.shape, Y_train.shape)

print('\nLoading test set')

df_test = pd.read_csv('pima_test.csv')
print(df_test.head())

X_test = df_test.values[:,0:8]
Y_test = df_test.values[:,8]
print(X_test.shape, Y_test.shape)

# preprocessing

print('\nStandardizing input variables')

scaler = StandardScaler()
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

# model training

print('Fitting logistic regression model')
model = LogisticRegression()
model.fit(X_train, Y_train)

# model evaluation

def evaluate_classifier(X, Y, model):
	
	Yhat = model.predict(X)

	accuracy = accuracy_score(Y, Yhat)
	print(f'Accuracy: {accuracy:.2f}')
	print(f'Misclassification error: {1-accuracy:.2f}')

	conf_matrix = confusion_matrix(Y, Yhat)
	print(f'Confusion matrix:\n{conf_matrix}')

	# extract components of the confusion matrix
	tn, fp, fn, tp = conf_matrix.ravel()

	# compute sensitivity, specificity, precision, NPV
	print(f'TPR: {tp/(tp+fn):.2f} (aka sensitivity, recall)')
	print(f'TNR: {tn/(tn+fp):.2f} (aka specificity)')
	print(f'PPV: {tp/(tp+fp):.2f} (aka precision)')
	print(f'NPV: {tn/(tn+fn):.2f}')

	# predict the probability for class 1 (not just class label)
	pr = model.predict_proba(X)[:,1]

	# calculate AUROC
	auroc = roc_auc_score(Y, pr)
	print(f'AUROC: {auroc:.2f}')

	return Yhat, pr

print('\nTrain evaluations')
Yhat_train, pr_train = evaluate_classifier(X_train, Y_train, model)

print('\nTest evaluations')
Yhat_test, pr_test = evaluate_classifier(X_test, Y_test, model)

# plot ROC and PR curves
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25))

# ROC curve 
ax = axes[0]
fpr, tpr, _ = roc_curve(Y_train, pr_train)
ax.plot(fpr, tpr, zorder=2, label='train')
fpr, tpr, _ = roc_curve(Y_test, pr_test)
ax.plot(fpr, tpr, zorder=2, label='test')
ax.plot([0, 1], [0, 1], zorder=1, label='random')
ax.legend(frameon=False, loc='lower right')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_title("ROC curve")

# PR curve
ax = axes[1]
precision, recall, _ = precision_recall_curve(Y_train, pr_train)
ax.plot(recall, precision, zorder=2, label='train')
precision, recall, _ = precision_recall_curve(Y_test, pr_test)
ax.plot(recall, precision, zorder=2, label='test')
ax.plot([0, 1], [0.5, 0.5], zorder=1, label='random')
ax.legend(frameon=False, loc='lower right')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title("PR curve")

for ax in axes:
	ax.set_axisbelow(True) # grid lines behind data
	ax.grid(True, linestyle=':', color='lightgray')
	ax.set_ylim(0, 1)
	ax.set_xlim(0, 1)

sns.despine(fig)
fig.tight_layout()
fig.savefig('log_reg_evals.png', bbox_inches='tight', dpi=400)
