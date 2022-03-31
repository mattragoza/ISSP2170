import numpy as np
import pandas as pd
import pickle

import bayes

data_file = 'data/training_data.csv'
model_file = 'model.pkl'

# load training data set

print(f'Loading training set from {data_file}')

data = pd.read_csv(data_file)
data[data == 0] = np.nan
print(data.describe().T)

X = data.values - 1
print(X.shape)

# model training

print('\nInitializing latent variable model')
model = bayes.NaiveBayes(p_X=[5,3,3,4,5,4], p_Y=4)
print(model)

print('\nFitting latent variable model')
model.fit(X)
print(model)


print(f'Saving model to {model_file}')

with open(model_file, 'wb') as f:
	pickle.dump(model, f)


print((X[:,5:] == np.arange(4)).sum(axis=0))
