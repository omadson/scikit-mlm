"""Minimal Learning Machine classes for regression and classification."""
import numpy as np
import scipy as sp
import pandas as pd
import itertools
from fcmeans import FCM
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

# UTILS functions used in some codes
# one-hot: convert output to one-hot encoding
def one_hot(y):
    y = [int(i) for i in y.tolist()]
    l = len(y)
    c = len(np.unique(y))
    y_oh = np.zeros((l, c+1))
    y_oh[np.arange(l), y] = 1
    return y_oh

# multiresponse sparse regression
def mrsr(X, T, norm=1, max_k=None):
    n, m = X.shape
    q    = T.shape[1] 
    c_k = np.zeros(m)
    W_k = np.zeros((m,q))
    A = set()
    Y_k     = X @ W_k

    max_k = range(m-1) if max_k == None else range(max_k)
    for k in max_k:
        # selecionar a coluna mais correlacionada   
        (T - Y_k).shape
        C_k     = (T - Y_k).T @ X
        c_k     = np.array([np.linalg.norm(C_k[:,j],1) for j in range(m)])
        c_k[list(A)] = 0
        c_k_hat = c_k.argmax()
        A       = A.union({c_k_hat})
        X_k     = X[:,list(A)]
        
        W_k_hat = np.linalg.pinv(X_k).dot(T)
        Y_k_hat = X_k @ W_k_hat
        
        W_k_hat_ = np.zeros((m,q))
        
        W_k_hat_[list(A),:] = W_k_hat
        
        # escolha do lb
        S = np.array(list(itertools.product([0, 1], repeat=q)))
        S[S == 0] = -1
        
        U_k = C_k.copy()
        V_k = (Y_k_hat - Y_k).T @ X
        lb = list()
        for j in set(range(m)).difference(A):
            u_kj = U_k[:,j]
            v_kj = V_k[:,j]
            LB = list()
            for i in range(S.shape[0]):
                s = S[i,:]
                LB.append((c_k_hat - s.T @ u_kj) / (c_k_hat - s.T @ v_kj))
            LB = np.array(LB)
            try:
                lb.append(LB[LB>0].min())
            except Exception as e:
                lb.append(np.abs(LB).min())
        
        lb_op = np.array(lb).min()
        
        Y_k = ((1 - lb_op) * Y_k) + (lb_op * Y_k_hat)
        W_k = ((1 - lb_op) * W_k) + (lb_op * W_k_hat_)
        
    return W_k,list(A)

# opposite neighborhood
def ON(X, y, neighborhood_size=1, D_in=None, D_out=None):
    if neighborhood_size <= X.shape[0]:
        D_in  = D_in  if isinstance(D_in, np.ndarray)  else sp.spatial.distance.cdist(X,X)
        D_out = D_out if isinstance(D_out, np.ndarray) else sp.spatial.distance.cdist(y,y)

        DD = D_out != 0
        opposite_neighborhood = set()
        for i in range(D_out.shape[0]):
            DDD = D_in[i,:].argsort()
            opposite_neighborhood = opposite_neighborhood.union(set(DDD[DD[i,DDD]][:neighborhood_size]))
        on_index = np.zeros(X.shape[0])
        on_index[list(opposite_neighborhood)] = 1
        if len(opposite_neighborhood) == X.shape[0]:
            return ON(X, y, neighborhood_size=neighborhood_size-1, D_in=D_in, D_out=D_in)
        else:
            return on_index == 1, D_in, D_out
    else:
        return ON(X, y, neighborhood_size=neighborhood_size-1, D_in=D_in, D_out=D_in)


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
        self.rp_y     = y[self.rp_index,:]

        # compute pairwise distance matrices
        #  - D_in: input space
        #  - D_out: output space
        self.D_in  = sp.spatial.distance.cdist(X,self.rp_X)
        self.D_out = sp.spatial.distance.cdist(y,self.rp_y)

        # compute the distance regression matrix using OLS
        self.B = np.linalg.pinv(self.D_in) @ self.D_out

        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "B")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return np.array([self.get_output(x)[0] for x in X])

    def get_output(self, x):
        J = lambda y: self.in_cost(y, x)
        out = sp.optimize.least_squares(J, x0=self.rp_y.mean(axis=0), method='lm')
        return out.x

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
        return (d_out**2 - (d_in @ self.B)**2)[0]

# MLM for classification: https://doi.org/10.1016/j.neucom.2014.11.073
class MLMC(MLMR):
    def fit(self, X, y=None):
        # convert outputs to one-hot encoding
        y = one_hot(y) if len(y.shape) == 1 else y
        # fit model using MLMR procedure
        return super().fit(X,y)

    def predict(self, X, y=None):
        try:
            getattr(self, "B")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        return np.array([np.argmax(self.get_output(x)) for x in X])

    


# nearest neighbor MLM (NN-MLM): https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec9
class NN_MLM(MLMC):
    def predict(self, X, y=None):
        try:
            getattr(self, "B")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        # compute matrix of distances from input RPs
        D_in = sp.spatial.distance.cdist(X,self.rp_X)
        # estimate matrix of distances from output RPs
        D_out_hat = D_in @ self.B

        return self.rp_y[D_out_hat.argmin(axis=1),:].argmax(axis=1)


# opposite neighborhood MLM (ON-MLM): https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2018-198.pdf
class ON_MLM(NN_MLM):
    def __init__(self, neighborhood_size=None):

        # number of reference points
        self.neighborhood_size = neighborhood_size

    def fit(self, X, y=None):
        # convert outputs to one-hot encoding
        y = one_hot(y) if len(y.shape) == 1 else y

        # opposite neighborhood procedure
        # first time
        on_index_1, D_in, D_out = ON(X, y, neighborhood_size=self.neighborhood_size)
        # second time
        on_index_2,_,_ = ON(X[~on_index_1,:],
                            y[~on_index_1,:],
                            neighborhood_size=1,
                            D_in=D_in[~on_index_1,:][:,~on_index_1],
                            D_out=D_out[~on_index_1,:][:,~on_index_1])

        self.rp_index = np.array([i for i, x in enumerate(~on_index_1) if x])[[i for i, x in enumerate(on_index_2) if x]]


        self.rp_X     = X[self.rp_index,:]
        self.rp_y     = y[self.rp_index,:]

        # compute pairwise distance matrices
        #  - D_in: input space
        #  - D_out: output space

        self.D_in  = D_in[:,self.rp_index] 
        self.D_out = D_out[:,self.rp_index] 

        # compute the distance regression matrix using OLS
        self.B = np.linalg.pinv(self.D_in) @ self.D_out

        return self



class w_MLM(NN_MLM):

    def fit(self, X, y=None):
        # convert outputs to one-hot encoding
        w = np.zeros(y.shape)
        labels = np.unique(y)
        for label in labels:
            w[y == label] = np.mean(y == label)
        y = one_hot(y) if len(y.shape) == 1 else y

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


        self.W = np.diag(w)
        # compute the distance regression matrix using OLS
        self.B = np.linalg.inv(self.D_in.T @ self.W @ self.D_in) @ self.D_in.T @ self.W @ self.D_out

        return self
        

class OS_MLM(NN_MLM):
    def __init__(self, max_rp_number=None):

        # number of reference points
        self.max_rp_number = max_rp_number

    def fit(self, X, y=None):
        # convert outputs to one-hot encoding
        y = one_hot(y) if len(y.shape) == 1 else y

        self.D_in  = sp.spatial.distance.cdist(X,X)
        self.D_out  = (y * (-1)) + 1

        self.B, self.rp_index = mrsr(self.D_in, self.D_out, norm=1, max_k=self.max_rp_number)
        # self.B = self.B[self.B != 0]

        self.B = self.B[self.rp_index,:]        

        self.rp_X = X[self.rp_index,:]
        self.rp_y = np.eye(y.shape[1])

        return self



class FCM_MLM(NN_MLM):
    def __init__(self, max_rp_number=None):

        # number of reference points
        self.max_rp_number = max_rp_number

    def fit(self, X, y=None):
        

        # 
        fcm = FCM(n_clusters=self.max_rp_number)
        fcm.fit(X)
        c = fcm.u.argmax(axis=1)
        homongenious_clusters = np.where(pd.DataFrame({'c': c, 'y': y}).groupby('c').mean().isin(np.unique(y)))[0]


        # convert outputs to one-hot encoding
        y = one_hot(y) if len(y.shape) == 1 else y

        

        self.rp_X  = fcm.centers[homongenious_clusters,:]
        self.rp_y = np.eye(y.shape[1])

        self.D_in  = sp.spatial.distance.cdist(X,self.rp_X)
        self.D_out  = (y * (-1)) + 1

        self.B = np.linalg.pinv(self.D_in) @ self.D_out
