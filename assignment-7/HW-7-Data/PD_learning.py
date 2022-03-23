import numpy as np
import pandas as pd
import pickle

import bayes

data_file = 'pneumonia.csv'
model_file = 'model.pkl'

# load training data set

print(f'Loading training set from {data_file}')

data = pd.read_csv(data_file)
print(data)

X = data.values[:,:4]
Y = data.values[:,4]
print(X.shape, data.columns[:4])
print(Y.shape, data.columns[4])

# model training

print('\nFitting naive Bayes model')
model = bayes.NaiveBayes(p_X='BBBB', p_Y='B')
model.fit(X, Y)

print(f'\nSaving model to {model_file}')

with open(model_file, 'wb') as f:
	pickle.dump(model, f)

print('\nPredicting probabilities')

phat = model.predict_proba(np.array([[1,0,1,0]]))
print(phat)

nan = np.nan
phat = model.predict_proba(np.array([[1,nan,1,nan]]))
print(phat)
