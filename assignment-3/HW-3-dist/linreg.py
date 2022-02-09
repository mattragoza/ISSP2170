import numpy as np


def quadratic_expansion(X):
    '''
    Expand data frame of predictors with
    all quadratic and interaction terms.

    Args:
        X: N x D data frame of predictors.
    Returns:
        N x K expanded data frame of predictors,
            where K = D + D(D + 1)/2.
    '''
    N, D = X.shape
    columns = list(X.columns)
    for i, col_i in enumerate(columns):
        for j, col_j in enumerate(columns[i:]):
            new_col = f'{col_i}_x_{col_j}'
            X[new_col] = X[col_i] * X[col_j]
    return X


class LinearRegression(object):

    def fit(
        self, X, Y,
        shuffle=False,
        batch=False,
        n_iters=1000,
        alpha_fn=lambda i, n_iters: 5e-2/i,
        init_fn=lambda shape: np.ones(shape),
    ):
        '''
        Fit a linear regression model using
        gradient descent.

        Args:
            X: N x D matrix of predictors.
            Y: Length N vector of outcomes.
            shuffle: Randomize data order each epoch.
            batch: Each update uses full data set.
            n_iters: Maximum number of iterations.
            alpha_fn: Learning rate schedule.
            init_fn: Weight initialization scheme.
        Returns:
            Length (n_iters+1) vector of iterations.
            Length (n_iters+1) vector of mean squared errors.
            Length (n_iters+1) vector of learning rates.
        '''
        # convert to arrays
        X = np.array(X)
        Y = np.array(Y)

        # check input shapes
        N, D = X.shape
        assert Y.shape == (N,)
        Y = Y.reshape(N, 1)

        # add intercept term
        ones = np.ones((N, 1))
        X = np.append(ones, X, axis=1)
        D += 1

        # initialize coefficients
        self.W = init_fn((D, 1))
        print(X.shape, self.W.shape, Y.shape)

        # initialize return values
        iters = np.arange(n_iters+1)
        alpha = alpha_fn(iters+1, n_iters+1)
        MSE = np.full(n_iters+1, np.nan)

        # perform gradient descent
        for i in iters:

            if batch: # batch mode
                if shuffle:
                    order = np.random.permutation(N)
                    X = X[order]
                    Y = Y[order]
                X_curr = X
                Y_curr = Y

            else: # online mode
                j = i%N
                if shuffle and j == 0:
                    order = np.random.permutation(N)
                    X = X[order]
                    Y = Y[order]
                X_curr = X[j:j+1]
                Y_curr = Y[j:j+1]

            # compute predictions and error
            Y_pred = X_curr @ self.W
            Y_diff = Y_curr - Y_pred
            MSE[i] = (Y_diff**2).mean()
            if i % 50 == 0:
                print(f'Iteration {i}\tMSE = {MSE[i]:.2f}\tlr = {alpha[i]:.6f}')

            if i == n_iters:
                break

            # compute error gradient
            grad_W = -2 * X_curr.T @ Y_diff

            # apply weight update
            self.W -= alpha[i] * grad_W

        return iters, MSE, alpha

    def predict(self, X):
        '''
        Use linear regression model to predict
        outcomes for the given predictors.

        Args:
            X: N x D matrix of predictors.
        Returns:
            Length N vector of predicted outcomes.
        '''
        # check input shape
        N = X.shape[0]
        D = self.W.shape[0]
        assert X.shape == (N, D-1)

        # add intercept term
        ones = np.ones((N, 1))
        X = np.append(ones, X, axis=1)

        return X @ self.W
