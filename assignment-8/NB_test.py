import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy import stats

import bayes

data_file = 'data/testing_data.csv'
model_file = 'model.pkl'

# load training data set

print(f'Loading test set from {data_file}')

data = pd.read_csv(data_file)
data = data.dropna() # empty rows
data[data == 0] = np.nan
print(data.describe().T)

X = data.values - 1
Y = X[:,5].copy()
X[:,5] = np.nan
print(X.shape, Y.shape)

# load model from file

print(f'\nLoading model from {model_file}')

with open(model_file, 'rb') as f:
	model = pickle.load(f)

print('\nPredicting probabilities')

# p[i,d] = P(X_6=d|X=x_i)
# p[i,d] = P(X_6=d,X=x_i) / ∑d P(X_6=d,X=x_i)
N, D = 75, 4
p = np.zeros((N, D))
for d in range(D):
	X_d = X.copy()
	X_d[:,5] = d
	p[:,d] = model.likelihood(X_d, nan=1.0)

p /= p.sum(axis=1, keepdims=True)
Y_hat = np.argmax(p, axis=1)

print('\nConfusion matrix')
conf_mat = confusion_matrix(Y, Y_hat)
print(conf_mat)

acc = accuracy_score(Y, Y_hat)
print(f'Accuracy = {acc:.2f}')
print(f'Error = {1-acc:.2f}')

print('\nCreating random predictor')
# random predictor
# error
# 1 - P(Y=Yhat)
# 1 - ∑j P(Y=j,Yhat=j)
# 1- ∑k

n = (Y[:,np.newaxis] == np.arange(4)).sum(axis=0)
print(n, n.sum())

p_test = n / n.sum()
print(p_test)

p_uniform = np.full(D, 1/D)
print(p_uniform)

print(p_test * p_uniform)
print(p_test.dot(p_uniform))
print(1 - p_test.dot(p_uniform))

print('\nPerforming chi-squared hypothesis test')

np.random.seed(0)
Y_null = np.random.randint(0, D, N)
test_mat = confusion_matrix(Y_null, Y_hat)

chi2, p, dof, _ = stats.chi2_contingency(test_mat)
print(f'chi2 = {chi2:.4f}, p = {p:.4f}, dof = {dof}')
