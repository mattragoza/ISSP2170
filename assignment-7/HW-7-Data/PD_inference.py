import numpy as np
import pandas as pd
import pickle

import bayes

data_file = 'example.csv'
model_file = 'model.pkl'

# load inference data set

print(f'Loading inference set from {data_file}')

data = pd.read_csv(data_file)
data[data == -1] = np.nan
print(data)

X = data.values[:,:4]
print(X.shape, data.columns[:4])

# load model from file

print(f'\nLoading model from {model_file}')

with open(model_file, 'rb') as f:
	model = pickle.load(f)

print('\nPredicting probabilities')

phat = model.predict_proba(X)
print(phat)
