import numpy as np
import pandas as pd
import pickle

import bayes

data_file = 'data/training_data.csv'
model_file = 'model.pkl'

# load training data set

print(f'Loading training set from {data_file}')

data = pd.read_csv(data_file)
print(data)

X = data.values
print(X.shape, data.columns)


# model training

print('\nFitting naive Bayes model')
model = bayes.NaiveBayes(p_X='CCCCCC', p_Y='C')
model.fit(X)

exit()
print(f'\nSaving model to {model_file}')

with open(model_file, 'wb') as f:
	pickle.dump(model, f)

print('\nPredicting probabilities')

phat = model.predict_proba(np.array([[1,0,1,0]]))
print(phat)

nan = np.nan
phat = model.predict_proba(np.array([[1,nan,1,nan]]))
print(phat)
