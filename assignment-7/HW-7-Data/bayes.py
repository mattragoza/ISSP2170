import numpy as np
from scipy import stats


class RandomVariable(object):
    '''
    A random variable from some family
    of probability distributions that
    can be fit to data using MLE.
    '''
    @classmethod
    def create(cls, rv_type):
        '''
        Factory method for creating random
        variables by type of distribution.

        Args:
            rv_type:
                B = Bernoulli distribution
                N = Normal distribution
                E = Exponential distribution
        Returns
            A new RandomVariable that follows
                the specified distribution.
        '''
        rv_type = rv_type.upper()
        if rv_type == 'B':
            return BernoulliVariable()
        elif rv_type == 'N':
            return NormalVariable()
        elif rv_type == 'E':
            return ExponentialVariable()
        else:
            raise ValueError(
                f'unknown distribution: {rv_type} (try "B", "N", or "E")'
            )

    def fit(self, X):
        '''
        Estimate variable parameters
        that maximize likelihood of X.

        Args:
            X: N vector of values.
        Returns:
            None.
        '''
        raise NotImplementedError

    def predict(self, X):
        '''
        Probability density at X.

        Args:
            X: N vector of values.
        Returns:
            N vector of probability
                density values at X.
        '''
        raise NotImplementedError


class BernoulliVariable(RandomVariable):
    '''
    A random variable that follows
    a Bernoulli distribution.
    '''
    def __init__(self, theta=0.5):
        assert 0 <= theta <= 1.0
        self.theta = theta

    def fit(self, X):
        assert set(X.flatten()) <= {0, 1}
        self.theta = np.mean(X)

    def predict(self, X, allow_missing=False):
        assert set(X[~np.isnan(X)].flatten()) <= {0, 1}
        return self.theta**X * (1-self.theta)**(1-X)

    def __repr__(self):
        return f'B(theta={self.theta:.2f})'


class NormalVariable(RandomVariable):
    '''
    A random variable that follows
    a univariate normal distribution.
    '''
    def __init__(self, mu=0.0, sigma=1.0):
        assert sigma > 0
        self.mu = mu
        self.sigma = sigma

    def fit(self, X):
        self.mu = np.mean(X)
        self.sigma = np.std(X, ddof=1) # unbiased

    def predict(self, X):
        return 1/(self.sigma*np.sqrt(2*np.pi)) * np.exp(
            -(X - self.mu)**2 / (2*self.sigma**2)
        )

    def __repr__(self):
        return f'N(mu={self.mu:.2f}, sigma={self.sigma:.2f})'


class ExponentialVariable(RandomVariable):
    '''
    A random variable that follows
    an exponential distribution.
    '''    
    def __init__(self, mu=1.0):
        assert mu > 0
        self.mu = mu

    def fit(self, X):
        assert all(X >= 0)
        self.mu = np.mean(X)

    def predict(self, X):
        assert all(X >= 0)
        return 1/self.mu * np.exp(-X/self.mu)

    def __repr__(self):
        return f'E(mu={self.mu:.2f})'


class NaiveBayes(object):
    '''
    A Naive Bayes classifier.
    '''
    def __init__(self, p_X, p_Y):
        '''
        Initialize the model by specifying families
        of probability distributions that conditional
        and prior densities should follow.

        Args:
            p_X: List of conditional density families.
            p_Y: Prior density family.
        '''
        assert p_Y.upper() == 'B', 'only Bernoulli prior is supported'

        # initialize conditional densities
        #   p_X[i][j] = p(X_i|Y=j)
        self.p_X = [
            [RandomVariable.create(x) for y in range(2)] for x in p_X
        ]

        # initialize prior density
        #   p_Y = p(Y)
        self.p_Y = RandomVariable.create(p_Y)

    def fit(self, X, Y):
        '''
        Estimate model parameters by MLE.

        Args:
            X: N x D matrix of attribute values.
            Y: N vector of class values.
        Returns:
            None.
        '''
        N, D = X.shape
        assert Y.shape == (N,)
        assert D == len(self.p_X)

        # estimate prior parameters
        self.p_Y.fit(Y)
        print(self.p_Y)

        # estimate conditional parameters
        for i in range(D):
            self.p_X[i][0].fit(X[Y==0,i])
            self.p_X[i][1].fit(X[Y==1,i])
            print(f'{i}\n\t{self.p_X[i][0]}\n\t{self.p_X[i][1]}')

    def predict_proba(self, X):
        '''
        Predict class posterior distribution.

        Args:
            X: N x D matrix of attribute values.
        Returns:
            N x 2 matrix of class probabilities.
        '''
        N, D = X.shape
        assert D == len(self.p_X)

        # class prior probabilities
        #   p_Y0 = p(Y=0)
        #   p_Y1 = p(Y=1)

        p_Y0 = self.p_Y.predict(np.zeros(N))
        p_Y1 = self.p_Y.predict(np.ones(N))

        # class conditional probabilities
        #   p_X_Y0 = p_X[i][0] = p(X_i|Y=0)
        #   p_X_Y1 = p_X[i][1] = p(X_i|Y=1)

        p_X_Y0 = np.zeros((N, D))
        p_X_Y1 = np.zeros((N, D))
        for i in range(D):
            p_X_Y0[:,i] = self.p_X[i][0].predict(X[:,i])
            p_X_Y1[:,i] = self.p_X[i][1].predict(X[:,i])

        # handle missing values by marginalization
        p_X_Y0 = np.where(np.isnan(X), 1.0, p_X_Y0)
        p_X_Y1 = np.where(np.isnan(X), 1.0, p_X_Y1)

        # joint probabilities
        #   p_XY0 = p(X,Y=0)
        #   p_XY1 = p(X,Y=1)

        p_XY0 = np.prod(p_X_Y0, axis=1) * p_Y0
        p_XY1 = np.prod(p_X_Y1, axis=1) * p_Y1

        # class posterior probabilities
        #   p_Y0_X = p(Y=0|X) = p(X,Y=0) / sum_y p(X,Y=y)
        #   p_Y1_X = p(Y=1|X) = p(X,Y=1) / sum_y p(X,Y=y)

        p_Y0_X = p_XY0 / (p_XY0 + p_XY1)
        p_Y1_X = p_XY1 / (p_XY0 + p_XY1)

        return np.stack([p_Y0_X, p_Y1_X], axis=1)

    def predict(self, X):
        '''
        Predict classes for given data.

        Args:
            X: N x D matrix of attribute values.
        Returns:
            N vector of class predictions.
        '''
        # predict class as argmax of posterior
        p_Y_X = self.predict_proba(X)
        return np.argmax(p_Y_X, axis=1)
