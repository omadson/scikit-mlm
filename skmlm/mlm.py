"""Minimal Learning Machine classes for regression and classification."""
import numpy as np
import scipy as sp

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


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
