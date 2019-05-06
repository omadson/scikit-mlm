"""Minimal Learning Machine classes for regression and classification."""
import numpy as np
import scipy as sp
import time

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

# MLM for regression: https://doi.org/10.1016/j.neucom.2014.11.073
class MLMR(BaseEstimator, RegressorMixin):
    def __init__(self, rp_number=None):

        # number of reference points
        self.rp_number = rp_number

    def fit(self, X, y=None):
        # random select of reference points for inputs and outputs
        if self.rp_number == None:
            self.rp_number = int(np.ceil(0.1 * X.shape[0]))
        self.rp_index = np.random.choice(X.shape[0], self.rp_number, replace=False)
        self.rp_X     = X[self.rp_index,:]
        self.rp_y     = y[self.rp_index]

        # compute pairwise distance matrices
        #  - D_in: input space
        #  - D_out: output space
        self.D_in  = sp.spatial.distance.cdist(X,self.rp_X)
        self.D_out = sp.spatial.distance.cdist(y[np.newaxis].T,self.rp_y[np.newaxis].T)

        # compute the distance regression matrix using OLS
        self.B = np.linalg.pinv(self.D_in).dot(self.D_out)

        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "B")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return np.array([self.get_output(x)[0] for x in X])

    def get_output(self, x):
        J = lambda y: self.in_cost(y, x)
        out = sp.optimize.least_squares(J, x0=self.rp_y.mean(), method='lm')
        return out.x

    def in_cost(self, y, x):
        """internal cost function"""
        # make y a vector
        y  = np.array([y])

        # compute pairwise distance vectors
        #  - d_in: input space
        #  - d_out: output space
        d_in  = sp.spatial.distance.cdist(x[np.newaxis],self.rp_X)
        d_out = sp.spatial.distance.cdist(y.T,self.rp_y[np.newaxis].T)

        # compute the internal cost function
        return (d_out**2 - d_in.dot(self.B)**2)[0]

# MLM for classification: https://doi.org/10.1016/j.neucom.2014.11.073
class MLMC(BaseEstimator, ClassifierMixin):
    def __init__(self, rp_number=None):

        # number of reference points
        self.rp_number = rp_number

    def one_hot(self, y):
        l = y.shape[0]
        c = len(np.unique(y))
        y_oh = np.zeros((l, c+1))
        y_oh[np.arange(l), y] = 1
        return y_oh

    def fit(self, X, y=None):
        # convert outputs to one-hot encoding
        y = self.one_hot(y) if len(y.shape) == 1 else y

        # random select of reference points for inputs and outputs
        if self.rp_number == None:
            self.rp_number = int(np.ceil(0.1 * X.shape[0]))
        self.rp_index = np.random.choice(X.shape[0], self.rp_number, replace=False)
        self.rp_X     = X[self.rp_index,:]
        self.rp_y     = y[self.rp_index,:]

        # compute pairwise distance matrices
        #  - D_in: input space
        #  - D_out: output space
        self.D_in  = sp.spatial.distance.cdist(X,self.rp_X)
        self.D_out = sp.spatial.distance.cdist(y,self.rp_y)

        # compute the distance regression matrix using OLS
        self.B = np.linalg.pinv(self.D_in).dot(self.D_out)

        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "B")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return np.array([self.get_output(x) for x in X])

    def get_output(self, x):
        J = lambda y: self.in_cost(y, x)
        out = sp.optimize.least_squares(J, x0=self.rp_y.mean(axis=0), method='lm')
        return np.argmax(out.x)

    def in_cost(self, y, x):
        """internal cost function"""
        # make y a vector
        y  = np.array([y])

        # compute pairwise distance vectors
        #  - d_in: input space
        #  - d_out: output space
        d_in  = sp.spatial.distance.cdist(x[np.newaxis],self.rp_X)
        d_out = sp.spatial.distance.cdist(y,self.rp_y)

        # compute the internal cost function
        return (d_out**2 - d_in.dot(self.B)**2)[0]


# nearest neighbor MLM (NN-MLM): https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec9
class NN_MLM(BaseEstimator, ClassifierMixin):
    def __init__(self, rp_number=None):

        # number of reference points
        self.rp_number = rp_number

    def one_hot(self, y):
        l = y.shape[0]
        c = len(np.unique(y))
        y_oh = np.zeros((l, c+1))
        y_oh[np.arange(l), y] = 1
        return y_oh

    def fit(self, X, y=None):
        # convert outputs to one-hot encoding
        y = self.one_hot(y) if len(y.shape) == 1 else y

        # random select of reference points for inputs and outputs
        if self.rp_number == None:
            self.rp_number = int(np.ceil(0.1 * X.shape[0]))
        self.rp_index = np.random.choice(X.shape[0], self.rp_number, replace=False)
        self.rp_X     = X[self.rp_index,:]
        self.rp_y     = y[self.rp_index,:]

        # compute pairwise distance matrices
        #  - D_in: input space
        #  - D_out: output space
        self.D_in  = sp.spatial.distance.cdist(X,self.rp_X)
        self.D_out = sp.spatial.distance.cdist(y,self.rp_y)

        # compute the distance regression matrix using OLS
        self.B = np.linalg.pinv(self.D_in).dot(self.D_out)

        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "B")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        y_hat = list()
        # compute matrix of distances from input RPs
        D_in = sp.spatial.distance.cdist(X,self.rp_X)
        # estimate matrix of distances from output RPs
        D_out_hat = D_in.dot(self.B)
        
        return self.rp_y[D_out_hat.argmin(axis=1),:].argmax(axis=1)
        