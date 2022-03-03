import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	roc_auc_score,
	roc_curve,
	precision_recall_curve,
	auc
)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')

dpi = 100

def print_config(args):
	s = str(vars(args))
	s = s.replace(', ', ',\n    ')
	s = '{\n    ' + s[1:-1] + '\n}'
	print(s)


def evaluate_classifier(X, y, model):

	yhat = model.predict(X)

	accuracy = accuracy_score(y, yhat)
	print(f'Accuracy: {accuracy:.3f}')
	print(f'Misclassification error: {1-accuracy:.3f}')

	conf_matrix = confusion_matrix(y, yhat)
	print(f'Confusion matrix:\n{conf_matrix}')

	# extract components of the confusion matrix
	tn, fp, fn, tp = conf_matrix.ravel()

	# compute sensitivity, specificity, precision, NPV
	print(f'TPR: {tp/(tp+fn):.3f} (aka sensitivity, recall)')
	print(f'TNR: {tn/(tn+fp):.3f} (aka specificity)')
	print(f'PPV: {tp/(tp+fp):.3f} (aka precision)')
	print(f'NPV: {tn/(tn+fn):.3f}')

	try: # predict the probability for class 1 (not just class label)
		pr = model.predict_proba(X)[:,1]
	except AttributeError:
		pr = model.decision_function(X)

	# calculate AUROC
	auroc = roc_auc_score(y, pr)
	print(f'AUROC: {auroc:.3f}')

	return yhat, pr


def plot_classifier(
	plot_file,
	y_train, yh_train, pr_train,
	y_test,  yh_test,  pr_test
):
	# plot ROC and PR curves
	fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25))

	# ROC curve 
	ax = axes[0]

	fpr, tpr, _ = roc_curve(y_train, pr_train)
	auroc =  auc(fpr, tpr)
	ax.plot(fpr, tpr, zorder=2, label=f'train (AUC = {auroc:.3f})')
	fpr, tpr, _ = roc_curve(y_test, pr_test)
	auroc =  auc(fpr, tpr)
	ax.plot(fpr, tpr, zorder=2, label=f'test (AUC = {auroc:.3f})')
	ax.plot([0, 1], [0, 1], zorder=1, label='random')
	ax.legend(frameon=False, loc='lower right')
	ax.set_xlabel('False positive rate')
	ax.set_ylabel('True positive rate')
	ax.set_title("ROC curve")

	# PR curve
	ax = axes[1]
	precision, recall, _ = precision_recall_curve(y_train, pr_train)
	auprc =  auc(recall, precision)
	ax.plot(recall, precision, zorder=2, label=f'train (AUC = {auprc:.3f})')
	precision, recall, _ = precision_recall_curve(y_test, pr_test)
	auprc =  auc(recall, precision)
	ax.plot(recall, precision, zorder=2, label=f'test (AUC = {auprc:.3f})')
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
	fig.savefig(plot_file, bbox_inches='tight', dpi=dpi)


def write_classifier(out_file, *cols):
	with open(out_file, 'w') as f:
		for vals in zip(*cols):
			f.write(' '.join(map(str, vals)) + '\n')
