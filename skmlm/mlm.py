"""Minimal Learning Machine classes for regression and classification."""
import numpy as np

from scipy.spatial.distance import cdist
from scipy.optimize import least_squares

from fcmeans import FCM
from mrsr import MRSR
from .utils import ON, one_hot

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


# MLM for regression: https://doi.org/10.1016/j.neucom.2014.11.073
class MLMR(BaseEstimator, RegressorMixin):
    def __init__(self, rp_number=None):

        # number of reference points
        self.rp_number = rp_number

    def fit(self, X, y=None):
        self.X = X
        self.y = y
        # random select of reference points for inputs and outputs
        if self.rp_number == None:
            self.rp_number = int(np.ceil(0.1 * X.shape[0]))
        self.rp_index = np.random.choice(X.shape[0], self.rp_number, replace=False)
        self.rp_X     = X[self.rp_index,:]
        self.rp_y     = y[self.rp_index,:]

        # compute pairwise distance matrices
        #  - D_in: input space
        #  - D_out: output space
        self.D_in  = cdist(X,self.rp_X)
        self.D_out = cdist(y,self.rp_y)

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
        out = least_squares(J, x0=self.rp_y.mean(axis=0), method='lm')
        return out.x

    def in_cost(self, y, x):
        """internal cost function"""
        # make y a vector
        y  = np.array([y])

        # compute pairwise distance vectors
        #  - d_in: input space
        #  - d_out: output space
        d_in  = cdist(x[np.newaxis],self.rp_X)
        d_out = cdist(y,self.rp_y)

        # compute the internal cost function
        return (d_out**2 - (d_in @ self.B)**2)[0]

# MLM for classification: https://doi.org/10.1016/j.neucom.2014.11.073
class MLMC(MLMR):
    def set_params(self, X, y, rp_index):
        self.X        = X
        self.y        = y
        self.rp_X     = X[rp_index,:]
        # self.rp_y     = y[rp_index,:]
        self.rp_y = np.eye(y.shape[1])
        self.rp_index = rp_index

    def fit(self, X, y=None):
        self.X = X
        self.y = y
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

    def plot(self,plt,X=None, y=None):
        X = X if X != None else self.X
        y = y if y != None else self.y
        y = y.argmax(axis=1) if len(y.shape) > 1 else y

        if X.shape[1] == 2:

            h = .005  # step size in the mesh
            # create a mesh to plot in
            x_min, x_max = X[:, 0].min(), X[:, 0].max()
            y_min, y_max = X[:, 1].min(), X[:, 1].max()
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))

            Z = self.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            fig = plt.figure(figsize=(6,6))
            plt.scatter(X[y == 0,0],X[y == 0,1], marker='o', c='orange')
            plt.scatter(X[y == 1,0],X[y == 1,1], marker='o', c='green')
            plt.scatter(self.rp_X[:,0],self.rp_X[:,1],alpha=0.9, facecolors='none',edgecolors='black',s=60,linewidths=2)
            plt.axis('off')
            plt.contour(xx, yy, Z, colors='black')


# nearest neighbor MLM (NN-MLM): https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec9
class NN_MLM(MLMC):
    def predict(self, X, y=None):
        try:
            getattr(self, "B")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        # compute matrix of distances from input RPs
        D_in = cdist(X,self.rp_X)
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

        rp_index = np.array([i for i, x in enumerate(~on_index_1) if x])[[i for i, x in enumerate(on_index_2) if x]]

        

        super().set_params(X,y,rp_index)

        # remove irrelevant columns form distance matrices
        self.D_in  = D_in[:,self.rp_index] 
        self.D_out = (y * (-1)) + 1

        # compute the distance regression matrix using OLS
        self.B = np.linalg.pinv(self.D_in) @ self.D_out

        return self


# weighted MLM (w_MLM): https://doi.org/10.1007/978-3-319-26532-2_61
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
        rp_index = np.random.choice(X.shape[0], self.rp_number, replace=False)
        
        super().set_params(X,y,rp_index)

        # compute pairwise distance matrices
        #  - D_in: input space
        #  - D_out: output space
        self.D_in  = cdist(X,self.rp_X)
        self.D_out = cdist(y,self.rp_y)


        self.W = np.diag(w)
        # compute the distance regression matrix using OLS
        self.B = np.linalg.inv(self.D_in.T @ self.W @ self.D_in) @ self.D_in.T @ self.W @ self.D_out

        return self
        
# optimally selected MLM (OS_MLM): https://doi.org/10.1007/978-3-030-03493-1_70
class OS_MLM(NN_MLM):
    def __init__(self, norm=1, feature_number=None,repetition_number=8,press=False,tol=10e-4):
        self.norm              = norm
        self.feature_number    = feature_number
        self.repetition_number = repetition_number
        self.tol               = tol
        self.press             = press

    def fit(self, X, y=None):
        # convert outputs to one-hot encoding
        y = one_hot(y) if len(y.shape) == 1 else y

        # compute pairwise distance matrices
        #  - D_in: input space
        #  - D_out: output space
        self.D_in  = cdist(X,X)
        self.D_out  = (y * (-1)) + 1

        self.feature_number = X.shape[0] if self.feature_number == None else self.feature_number

        mrsr = MRSR(norm=self.norm, feature_number=self.feature_number,repetition_number=self.repetition_number,press=self.press,tol=self.tol)
        mrsr.fit(self.D_in, self.D_out)

        super().set_params(X,y,mrsr.order)

        self.B = mrsr.W
        # self.B = np.linalg.pinv(self.D_in[:,self.rp_index]) @ self.D_out
        self.error = mrsr.corr

        return self

# fuzzy C-means MLM (FCM_MLM): https://doi.org/10.1007/978-3-319-95312-0_34
class FCM_MLM(NN_MLM):
    def __init__(self, max_rp_number=None):
        # number of reference points
        self.max_rp_number = max_rp_number

    def fit(self, X, y=None):
        self.X = X
        self.y = y
        # 
        fcm = FCM(n_clusters=self.max_rp_number)
        fcm.fit(X)
        c = fcm.u.argmax(axis=1)
        # homongenious_clusters = np.where(pd.DataFrame({'c': c, 'y': y}).groupby('c').mean().isin(np.unique(y)))[0]
        homongenious_clusters = np.where(np.bincount(np.unique(np.vstack((c,y)), axis=1)[0,:]) == 1)[0]


        # convert outputs to one-hot encoding
        y = one_hot(y) if len(y.shape) == 1 else y

        

        self.rp_X  = fcm.centers[homongenious_clusters,:]
        self.rp_y = np.eye(y.shape[1])

        self.D_in  = cdist(X,self.rp_X)
        self.D_out  = (y * (-1)) + 1

        self.B = np.linalg.pinv(self.D_in) @ self.D_out
        return self

# ℓ1/2-norm regularization MLM (L12_MLM): https://doi.org/10.1109/BRACIS.2018.00043
class L12_MLM(NN_MLM):
    def __init__(self, alpha=0.9, lb=0.1):
        # number of reference points
        self.alpha = alpha
        self.lb    = lb

    def fit(self, X, y=None):
        self.X = X
        self.y = y
        # convert outputs to one-hot encoding
        y = one_hot(y) if len(y.shape) == 1 else y

        # compute distance matrices with all data as RP
        self.rp_X  = X
        self.D_in  = cdist(X,self.rp_X)
        self.D_out = (y * (-1)) + 1


        # descend gradient setup
        epochs = 2000
        eta    = 0.01

        # Initialize the matrix B with values close to zero
        B_t = 0.001 * np.random.randn(self.D_in.shape[1],self.D_out.shape[1])

        # e = np.zeros(epochs)
        # descend gradient loop
        c = 0
        for t in range(epochs):
            # compute the Jacobian associated with the \ell_{1/2}-regularizer
            # BB   = np.sqrt(np.abs(B_t))
            DB_t = (1/2) * np.multiply(np.sign(B_t),1/np.sqrt(np.abs(B_t)))

            # compute the Jacobian of the loss function
            # E   = self.D_out - self.D_in @ B_t
            JB_t = (2 * self.D_in.T @ (self.D_out - self.D_in @ B_t)) + (self.lb * DB_t)

            # Update B_t with gradient descent rule
            B_t = B_t + eta * (JB_t)/(np.linalg.norm(JB_t,'fro'))
            
            # pruning phase
            c = c + 1
            if t >= 0.1 * epochs and c > 0.1 * epochs and t <= 0.7 * epochs:
                c = 0
                # compute the pruning threshold (gamma)
                B_t_norm = np.linalg.norm(B_t,axis=1)
                gamma = self.alpha * B_t_norm.mean()

                # create the list of the less important RPs
                no_pruning = ~(B_t_norm < gamma)

                # update matrices
                B_t = B_t[no_pruning,:]
                self.D_in = self.D_in[:,no_pruning]
                self.rp_X = self.rp_X[no_pruning,:]

            # e[t] = np.trace(E @ E.T) + self.lb * BB.sum()

        self.B = B_t
        self.rp_y = np.eye(y.shape[1])
        return self