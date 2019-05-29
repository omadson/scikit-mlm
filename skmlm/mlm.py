"""Minimal Learning Machine classes for regression and classification."""
import numpy as np

from scipy.spatial.distance import cdist
from scipy.optimize import least_squares

from fcmeans import FCM
from .mrsr import MRSR
from .utils import ON, pinv_, one_hot, ERRORS

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer


errors = ERRORS()

# MLM for regression (MLM): https://doi.org/10.1016/j.neucom.2014.11.073
class MLM(BaseEstimator, RegressorMixin):
    def __init__(self, rp_number=None):
        # number of reference points
        self.rp_number = rp_number
        #    if None, set rp_number to 10% of samples,
        #    if rp_number in [0,1], use as percentual.
        if self.rp_number == None: self.rp_number = 0.1
        
    def select_RPs(self):
        # random selection
        #    if <rp_number> equals to <N> use all points of RPs,
        #    else, select <rp_number> points at random.
        N = self.X.shape[0]

        if self.rp_number <= 1:    self.rp_number = int(self.rp_number * N)

        if self.rp_number == N:
            rp_id     = np.arange(N)
        else:
            rp_id     = np.random.choice(N, self.rp_number, replace=False)

        self.rp_X     = self.X[rp_id,:]
        self.rp_y     = self.y[rp_id,:]

        self.D_x = cdist(self.X,self.rp_X)
        self.D_y = cdist(self.y,self.rp_y)

    def fit_B(self):
        self.B        = pinv_(self.D_x) @ self.D_y

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.select_RPs()
        self.fit_B()
        return self

    def predict(self, X, y=None):
        erros.not_train()
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
class MLMC(MLM):
    def fit(self, X, y=None):
        # convert outputs to one-hot encoding
        y = self.oh_convert(y)
        # fit model using MLM procedure
        super().fit(X,y)

    def predict(self, X, y=None):
        if self.y_oh:
            return super().predict(X,y)
        else:
            return super().predict(X,y).argmax(axis=1)

    def oh_convert(self, y):
        self.y_oh = False if len(y.shape) == 1 else True
        if self.y_oh == False: y = one_hot(y)
        return y

    def plot(self,plt,X=None, y=None, figsize=(6,6)):
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
            fig = plt.figure(figsize=figsize)
            plt.scatter(X[y == 0,0],X[y == 0,1], marker='o', c='orange')
            plt.scatter(X[y == 1,0],X[y == 1,1], marker='o', c='green')
            plt.scatter(self.rp_X[:,0],self.rp_X[:,1],alpha=0.9, facecolors='none',edgecolors='black',s=60,linewidths=2)
            plt.axis('off')
            plt.contour(xx, yy, Z, colors='black')
            plt.show()
        else:
            print("X have more that two dimensions.")


# nearest neighbor MLM (NN-MLM): https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec9
class NN_MLM(MLMC):
    def predict(self, X, y=None):
        errors.not_train(self)
        # compute matrix of distances from input RPs
        D_in = cdist(X,self.rp_X)
        # estimate matrix of distances from output RPs
        D_out_hat = D_in @ self.B

        if self.y_oh:
            return self.rp_y[D_out_hat.argmin(axis=1),:]
        else:
            return self.rp_y[D_out_hat.argmin(axis=1),:].argmax(axis=1)


# opposite neighborhood MLM (ON-MLM): https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2018-198.pdf
class ON_MLM(NN_MLM):
    def __init__(self, neighborhood_size=None):
        # size of first neighborhood
        self.neighborhood_size = neighborhood_size

    def select_RPs(self):
        # select output RPs and compute output distance matrix
        self.rp_y = np.eye(self.y.shape[1])
        self.D_y  = (self.y * (-1)) + 1

        # opposite neighborhood procedure
        # first time
        on_index_1, self.D_x,D_y  = ON(self.X, self.y, neighborhood_size=self.neighborhood_size)
        # second time
        on_index_2,_,_ = ON(self.X[~on_index_1,:],
                            self.y[~on_index_1,:],
                            neighborhood_size=1,
                            D_x=self.D_x[:,~on_index_1][~on_index_1,:],
                            D_y=     D_y[:,~on_index_1][~on_index_1,:])

        rp_id  = np.array([i for i, x in enumerate(~on_index_1) if x])[[i for i, x in enumerate(on_index_2) if x]]
        # rp_id = on_index_1
        # define input reference points
        self.rp_X = self.X[rp_id,:]
        # remove irrelevant columns from distance matrices
        self.D_x  = self.D_x[:,rp_id]


# weighted MLM (w_MLM): https://doi.org/10.1007/978-3-319-26532-2_61
class w_MLM(NN_MLM):
    def fit_B(self):
        # create weight matrix
        y = self.y.argmax(axis=1)
        w = np.zeros(y.shape[0])
        labels = np.unique(y)
        for label in labels:
            w[y == label] = np.mean(y == label)

        self.W = np.diag(w)
        # compute the distance regression matrix using OLS
        self.B = np.linalg.inv(self.D_x.T @ self.W @ self.D_x) @ self.D_x.T @ self.W @ self.D_y

        
# optimally selected MLM (OS_MLM): https://doi.org/10.1007/978-3-030-03493-1_70
class OS_MLM(NN_MLM):
    def __init__(self, norm=1, feature_number=None,repetition_number=8,press=False,tol=None, pinv=False):
        self.norm              = norm
        self.feature_number    = feature_number

        if self.feature_number == None: self.feature_number = 0.1


        self.repetition_number = repetition_number
        self.tol               = tol
        self.press             = press
        self.pinv              = pinv

    def select_RPs(self):
        # convert outputs to one-hot encoding
        # self.y = self.oh_convert(self.y)
        # compute pairwise distance matrices
        #  - D_x: input space
        #  - D_y: output space
        D_x  = cdist(self.X,self.X)
        self.D_y  = (self.y * (-1)) + 1

        if self.feature_number <= 1:    self.feature_number = int(self.feature_number * self.X.shape[0])

        mrsr = MRSR(norm=self.norm,
                    feature_number=self.feature_number,
                    repetition_number=self.repetition_number,
                    press=self.press,
                    tol=self.tol,
                    pinv=self.pinv)

        mrsr.fit(D_x, self.D_y)

        rp_id = mrsr.order

        self.rp_X     = self.X[rp_id,:]
        self.rp_y     = self.y[rp_id,:]

        self.B = mrsr.W

    def fit_B(self): pass

# fuzzy C-means MLM (FCM_MLM): https://doi.org/10.1007/978-3-319-95312-0_34
class FCM_MLM(NN_MLM):
    def select_RPs(self):
        # random selection
        #    if <rp_number> equals to <N> use all points of RPs,
        #    else, select <rp_number> points at random.
        N = self.X.shape[0]

        if self.rp_number <= 1:    self.rp_number = int(self.rp_number * N)

        fcm = FCM(n_clusters=self.rp_number)
        fcm.fit(self.X)
        c = fcm.u.argmax(axis=1)
        # homongenious_clusters
        homongenious_clusters = np.where(np.bincount(np.unique(np.vstack((c,self.y.argmax(axis=1))), axis=1)[0,:]) == 1)[0]
        # get most closest samples from centers
        rp_id = cdist(fcm.centers[homongenious_clusters,:],self.X).argmin(axis=1)

        self.rp_X     = self.X[rp_id,:]
        self.rp_y     = self.y[rp_id,:]

        self.D_x = cdist(self.X,self.rp_X)
        self.D_y = cdist(self.y,self.rp_y)

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

# norm 2 regularization:
class L2_MLM(NN_MLM):
    def __init__(self, rp_number=None, C=1):
        # number of RPs and the regularization parameter
        self.rp_number = rp_number
        self.C         = C
    def fit_B(self):
        # compute the distance regression matrix using OLS
        self.B = np.linalg.inv(self.D_x.T @ self.D_x + self.C * np.eye(self.rp_X.shape[0])) @ self.D_x.T @ self.D_y