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
        if isinstance(rv_type, str):
            rv_type = rv_type.upper()
        if rv_type == 'B':
            return BernoulliVariable()
        elif rv_type == 'N':
            return NormalVariable()
        elif rv_type == 'E':
            return ExponentialVariable()
        try:
            return CategoricalVariable(k=int(rv_type))
        except:
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


class DiscreteVariable(RandomVariable):

    @property
    def support(self):
        return NotImplemented

    def get_dummy(self, X, nan=np.nan):
        assert X.ndim == 1
        missing = np.isnan(X)
        k = len(self.support)
        nax = np.newaxis
        X = (X[:,nax] == np.arange(k)[nax,:]).astype(float)
        X[missing] = nan
        return X


class ContinuousVariable(RandomVariable):

    @property
    def support(self):
        raise ValueError('cannot enumerate support of continuous variable')


class BernoulliVariable(DiscreteVariable):
    '''
    A random variable that follows
    a Bernoulli distribution.
    '''
    def __init__(self, theta=None):
        if theta is None:
            theta = np.random.uniform(0, 1)
        assert 0 <= theta <= 1
        self.theta = theta

    @property
    def support(self):
        return {0, 1}

    def fit(self, X):
        assert X.ndim in {1, 2}
        if X.ndim == 1:
            assert set(X) <= self.support
            self.theta = np.mean(X)
        elif X.ndim == 2:
            assert X.shape[1] == 2
            assert (X >= 0).all()
            assert np.allclose(X.sum(axis=1), 1.0)
            self.theta = np.mean(X, axis=0)[1]

    def predict(self, X):
        assert X.ndim == 1
        missing = np.isnan(X)
        assert set(X[~missing]) <= self.support
        return self.theta**X * (1-self.theta)**(1-X)

    def __repr__(self):
        return f'B(theta={self.theta:.2f})'


class CategoricalVariable(DiscreteVariable):
    '''
    A random variable that follows
    a Categorical distribution.
    '''
    def __init__(self, k, theta=None):

        if theta is None:
            theta = np.random.dirichlet(np.ones(k))
        else:
            theta = np.array(theta)

        assert theta.ndim == 1
        assert len(theta) == k
        assert all(theta >= 0.0)
        assert np.isclose(sum(theta), 1.0)

        self.k = k
        self.theta = theta

    @property
    def support(self):
        return {i for i in range(self.k)}

    def fit(self, X, weights=None):
        assert X.ndim in {1, 2}
        if X.ndim == 1:
            assert set(X) <= self.support
            self.theta = np.average(np.eye(self.k)[X], axis=0, weights=weights)
        elif X.ndim == 2:
            assert X.shape[1] == self.k
            assert (X >= 0.0).all()
            assert np.allclose(X.sum(axis=1), 1.0)
            self.theta = np.average(X, axis=0, weights=weights)

    def predict(self, X):
        assert X.ndim == 1
        missing = np.isnan(X)
        assert set(X[~missing]) <= self.support
        X_ind = self.get_dummy(X)
        X_ind[missing,:] = np.nan
        nax = np.newaxis
        return np.prod(self.theta[nax,:]**X_ind, axis=1)

    def __repr__(self):
        return f'C(k={self.k}, theta={self.theta})'


class NormalVariable(ContinuousVariable):
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


class ExponentialVariable(ContinuousVariable):
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
        # initialize prior distribution
        #   p_Y = p(Y)
        self.p_Y = RandomVariable.create(p_Y)
        self.K = len(self.p_Y.support)

        # initialize conditional distributions
        #   p_X[j][k] = p(X_k|Y=j)
        self.p_X = [
            [RandomVariable.create(x) for x in p_X] for y in self.p_Y.support
        ]
        self.D = len(p_X)

    def __str__(self):
        s = str(self.p_Y) + '\n'
        for j in self.p_Y.support:
            s += str(j) + '\n'
            for k in range(self.D):
                s += '\t' + str(self.p_X[j][k]) + '\n'
        return s

    def fit(self, X, Y=None, max_iter=1000, eps=1e-4):
        '''
        Estimate model parameters by MLE or EM.

        Args:
            X: N x D matrix of attribute values.
            Y: N vector of class values. If not provided,
                assume that class is latent and use EM.
        '''
        N, D = X.shape
        assert D == self.D
        K = self.K

        if Y is None:
            Y = np.full(N, np.nan)

        assert Y.shape == (N,)
        Y_missing = np.isnan(Y)
        Y = self.p_Y.get_dummy(Y)

        # use EM algorithm
        nax = np.newaxis
        ll = np.nan
        for i in range(max_iter):
            ll_prev = ll
            ll = 0

            # E-step: replace missing with expected values
            # p_Y_X[i,j] = p(Y=j|X=x_i) = π_j
            p_Y = self.class_prior_prob()
            p_Y_X = self.class_posterior_prob(X, nan=1.0)
            Y[Y_missing] = p_Y_X[Y_missing]
            ll += (Y * np.log(p_Y[nax,:])).sum()

            # E-step: update maximum likelihood estimates
            self.p_Y.fit(p_Y_X)

            # p_X_Y[d,j,k] = p(X_k=d|Y=j) = θ_kjd
            p_X_Y = self.class_conditional_prob()

            for j in range(K): # class index
                Y_j = Y[:,j]

                for k in range(D): # attribute index

                    # E-step: replace missing with expected values
                    p_X = self.p_X[j][k]
                    d_k = len(p_X.support)
                    X_k = p_X.get_dummy(X[:,k])
                    X_k_missing = np.isnan(X[:,k])
                    X_k[X_k_missing] = p_X_Y[nax,:d_k,j,k]

                    # M-step: update maximum likelihood estimates
                    self.p_X[j][k].fit(X_k, weights=Y_j+eps)

                    # p_X_Y[d,j,k] = p(X_k=d|Y=j) = θ_kjd
                    # X_k[i,d] = P(X_k=d|Y=j,θ) if missing, else one hot
                    # Y[i,j] * X_k[i,d] = P(X_k=d,Y=j|X=x_i,θ)
                    ll += (Y_j[:,nax] * X_k * np.log(p_X_Y[nax,:d_k,j,k])).sum()

            if np.isnan(ll):
                print(self)
            assert not np.isnan(ll)

            delta_ll = ll - ll_prev
            print(f'[Iteration {i+1}] ll = {ll:.4f} ({delta_ll:.4f})')
            if abs(delta_ll) < eps:
                break

    def mle_estimate(self, X, Y):
        '''
        Estimate model parameters that
        maximize likelihood of data.
        '''
        N, D = X.shape
        assert D == self.D
        assert Y.shape == (N,)
        K = self.K

        # estimate class prior parameters
        self.p_Y.fit(Y)

        # estimate class conditional parameters
        for j in range(K):
            for k in range(D):
                self.p_X[j][k].fit(X[Y==j,k])

    def class_prior_prob(self, Y=None):
        '''
        Get class prior probabilities.

        Args:
            Y: K vector of class indices, or None
                to return the full class distribution.
        Returns:
            p_Y: K vector of class prior probabilities,
                where p_Y[j] = P(Y=y_j).
        '''
        if Y is None:
            Y = np.arange(self.K)
        return self.p_Y.predict(Y)

    def class_conditional_prob(self, X=None, nan=np.nan):
        '''
        Get class conditional probabilities.

        Args:
            X: N x D matrix of attribute values, or None
                to return the full conditional distribution.
        Returns:
            p_X_Y: N x K x D class conditional probabilities,
                where p_X_Y[i,j,k] = p(X_k=x_ik|Y=j)
        '''
        if X is None:
            X = []
            M = max(len(p_X.support) for p_X in self.p_X[0])
            for k, p_X in enumerate(self.p_X[0]):
                X.append(np.arange(M).astype(float))
                X[k][len(p_X.support):] = np.nan
            X = np.stack(X, axis=1)

        N, D = X.shape
        assert D == self.D
        K = self.K

        p_X_Y = np.zeros((N, K, D))
        for j in range(K):
            for k in range(D):
                p_X_Y[:,j,k] = self.p_X[j][k].predict(X[:,k])

        # handle missing values
        p_X_Y = np.where(np.isnan(p_X_Y), nan, p_X_Y)

        return p_X_Y

    def joint_prob(self, X, nan=np.nan):
        '''
        Get joint probability density.

        Args:
            X: N x D matrix of attribute values.
        Returns:
            p_XY: N x K matrix of joint probabilities,
                where p_XY[i,j] = p(X=x_i,Y=j)
        '''
        p_Y = self.class_prior_prob()
        p_X_Y = self.class_conditional_prob(X, nan=nan)
        return np.prod(p_X_Y, axis=2) * p_Y[np.newaxis,:]

    def class_posterior_prob(self, X, nan=np.nan):
        '''
        Get class posterior distribution.

        Args:
            X: N x D matrix of attribute values.
        Returns:
            p_Y_X: N x K matrix of class probabilities,
                where p_Y_X[i,j] = p(Y=j|X=x_i)
        '''
        p_XY = self.joint_prob(X, nan=nan)
        return p_XY / p_XY.sum(axis=1, keepdims=True)



    def predict(self, X):
        '''
        Predict classes for given data.

        Args:
            X: N x D matrix of attribute values.
        Returns:
            Yhat: N vector of class predictions.
        '''
        p_Y_X = self.class_posterior_prob(X)
        return np.argmax(p_Y_X, axis=1)

    def likelihood(self, X, nan=np.nan):
        '''
        Get data distribution.

        Args:
            X: N x D matrix of attribute values.
        Returns:
            p_X: N vector of data probabilities.
        '''
        p_XY = self.joint_prob(X, nan=nan)
        return p_XY.sum(axis=1)
