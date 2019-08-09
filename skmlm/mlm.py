"""Minimal Learning Machine classes for regression and classification."""
import numpy as np
from scipy import fftpack
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares

# from fcmeans import FCM
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans as FCM
from mrsr import MRSR
from .utils import ON, pinv_, one_hot, ERRORS

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import mode
errors = ERRORS()

# MLM for regression (MLM): https://doi.org/10.1016/j.neucom.2014.11.073
class MLM(BaseEstimator, RegressorMixin):
    def __init__(self, rp_number=None, random_state=42):
        # random state
        self.random_state = random_state
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
            r = np.random.RandomState(self.random_state)
            rp_id     = r.choice(N, self.rp_number, replace=False)

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
        self.X_red = 1 - self.B.shape[0] / self.X.shape[0]
        self.y_red = 1 - self.B.shape[1] / self.y.shape[0]
        delattr(self, 'X')
        delattr(self, 'y')
        delattr(self, 'D_x')
        return self

    def predict(self, X, y=None):
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
        d_x  = cdist(x[np.newaxis],self.rp_X)
        d_y  = cdist(y,self.rp_y)

        # compute the internal cost function
        # print(((d_y**2 - (d_x @ self.B)**2) / np.abs(d_y))[0])
        return ((d_y**2 - (d_x @ self.B)**2)**2)[0]

    def plot(self,plt,X=None, y=None, figsize=None):
        # X = X if X != None else self.X
        # y = y if y != None else self.y

        X_ = np.linspace(X.min(), X.max(), 300)[np.newaxis].T
        y_ = self.predict(X_)

        if X.shape[1] == 1:
            fig = plt.figure(figsize=figsize) if figsize != None else plt.figure()
            plt.scatter(X,y, marker='o', c='orange')
            plt.scatter(self.rp_X[:,0],self.rp_y[:,0],alpha=0.9, facecolors='none',edgecolors='black',s=60,linewidths=2)
            plt.plot(X_, y_, c='black')
        else:
            print("X have more that one dimensions.")

# cubic equation MLM (C_MLM): https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec10
class C_MLM(MLM):
    def predict(self, X, y=None):
        errors.not_train(self)
        # compute matrix of distances from input RPs
        D_x = cdist(X,self.rp_X)
        # estimate matrix of distances from output RPs
        D_y_hat = D_x @ self.B

        
        a_i = self.rp_y.shape[0]
        b_i = -3 * self.rp_y.sum()

        RP_y = np.repeat(self.rp_y,X.shape[0],axis=1).T

        c = np.sum(3*RP_y**2 - D_y_hat**2, axis=1)
        d = np.sum(RP_y * D_y_hat**2 -  RP_y**3, axis=1)

        return np.array([self.roots_verify(np.roots([a_i, b_i,c[i], d[i]])) for i in range(len(d))])
    
    def roots_verify(self, roots):
        real_roots = []
        for root in roots:
            if np.isreal(root):
                real_roots.append(np.real(root))
        if len(real_roots) == 1:
            return real_roots[0]
        else:
            v = []
            for real_root in real_roots:
                v.append(np.sum((real_root - self.rp_y.T)**2 - cdist([[real_root]],self.rp_y)**2))
            return real_roots[np.array(v).argmin()]

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
        D_x = cdist(X,self.rp_X)
        # estimate matrix of distances from output RPs
        D_y_hat = D_x @ self.B

        if self.y_oh:
            return self.rp_y[D_y_hat.argmin(axis=1),:]
        else:
            return self.rp_y[D_y_hat.argmin(axis=1),:].argmax(axis=1)


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
    def __init__(self, norm=1, max_feature_number=None, pinv=False):
        self.norm               = norm
        self.max_feature_number = max_feature_number
        self.pinv              = pinv
        
        if self.max_feature_number == None: self.max_feature_number = 0.20
        

    def select_RPs(self):
        # convert outputs to one-hot encoding
        # self.y = self.oh_convert(self.y)
        # compute pairwise distance matrices
        #  - D_x: input space
        #  - D_y: output space
        self.D_x  = cdist(self.X,self.X)
        self.D_y  = (self.y * (-1)) + 1

        if self.max_feature_number <= 1:    self.max_feature_number = int(self.max_feature_number * self.X.shape[0])

        mrsr = MRSR(norm=self.norm,
                    max_feature_number=self.max_feature_number,
                    pinv=self.pinv)

        mrsr.fit(self.D_x, self.D_y)

        rp_id = mrsr.order

        self.rp_X     = self.X[rp_id,:]
        self.rp_y     = np.eye(self.y.shape[1])

        self.B = mrsr.W
        self.error = mrsr.error

    def fit_B(self): pass

# fuzzy C-means MLM (FCM_MLM): https://doi.org/10.1007/978-3-319-95312-0_34
class FCM_MLM(NN_MLM):
    def select_RPs(self):
        N = self.X.shape[0]

        if self.rp_number <= 1:    self.rp_number = int(self.rp_number * N)

        # fcm = FCM(n_clusters=self.rp_number, random_state=self.random_state, max_iter=50)
        fcm = FCM(k=self.rp_number, random_state=self.random_state, max_iter=50)
        fcm.fit(self.X)
        # c = fcm.u.argmax(axis=1)
        c = fcm.labels_
        # homongenious_clusters
        homongenious_clusters = np.where(np.bincount(np.unique(np.vstack((c,self.y.argmax(axis=1))), axis=1)[0,:]) == 1)[0]
        # centers = fcm.centers[homongenious_clusters,:]
        centers = fcm.cluster_centers_[homongenious_clusters,:]
        # if all clusters are heterogenious, use all clusters
        if len(centers) == 0:
            # centers = fcm.centers
            centers = fcm.cluster_centers_

        # get most closest samples from centers
        rp_id = cdist(centers,self.X).argmin(axis=1)

        self.rp_X     = self.X[rp_id,:]
        self.rp_y     = self.y[rp_id,:]

        self.D_x = cdist(self.X,self.rp_X)
        self.D_y = cdist(self.y,self.rp_y)

# ℓ1/2-norm regularization MLM (L12_MLM): https://doi.org/10.1109/BRACIS.2018.00043
class L12_MLM(NN_MLM):
    def __init__(self, alpha=0.7, lb=0.1, epochs=2000, eta=0.01, rp_min_size=0.05, random_state=42):
        # number of reference points
        self.alpha  = alpha
        self.lb     = lb
        self.epochs = epochs
        self.eta    = eta
        self.rp_min_size = rp_min_size
        self.random_state = random_state

    def select_RPs(self):
        # compute distance matrices with all data as RP
        rp_X = self.X
        rp_y = self.y
        D_x  = cdist(self.X,rp_X)
        D_y  = (-1) * self.y + 1

        # compute the minimun number of input reference points
        N = self.X.shape[0]
        if self.rp_min_size <= 1:    self.rp_min_size = int(self.rp_min_size * N)
        if self.rp_min_size > N:     self.rp_min_size = N

        # Initialize the matrix B with values close to zero
        r = np.random.RandomState(self.random_state)
        B_t = r.normal(0,0.001, (D_x.shape[1],D_y.shape[1]))
        # B_t = 0.001 * np.random.randn(D_x.shape[1],D_y.shape[1])

        # e = np.zeros(self.epochs)
        # descend gradient loop
        c = 0
        t = 0
        for t in range(self.epochs):
            # compute the Jacobian associated with the \ell_{1/2}-regularizer
            # BB   = np.sqrt(np.abs(B_t))
            DB_t = (1/2) * np.multiply(np.sign(B_t),1/np.sqrt(np.abs(B_t)))

            # compute the Jacobian of the loss function
            # E   = D_y - D_x @ B_t
            JB_t = (2 * D_x.T @ (D_y - D_x @ B_t)) + (self.lb * DB_t)

            # Update B_t with gradient descent rule
            B_t = B_t + self.eta * (JB_t)/(np.linalg.norm(JB_t,'fro'))
            
            # pruning phase
            c += 1
            if t >= 0.1 * self.epochs and c > 0.1 * self.epochs and t <= 0.7 * self.epochs:
                c = 0
                # compute the pruning threshold (gamma)
                B_t_norm = np.linalg.norm(B_t,axis=1)
                B_t_norm_mean = B_t_norm.mean()
                gamma = self.alpha * B_t_norm_mean
                # create the list of the less important RPs
                no_pruning = ~(B_t_norm < gamma)
                # check whether the number of remaining reference points exceeds the minimum number
                while (B_t[no_pruning,:].shape[0] < self.rp_min_size):
                    # update alpha to a new tiny value 
                    self.alpha = 0.5 * self.alpha
                    # compute the new pruning threshold (gamma)
                    gamma      = self.alpha * B_t_norm_mean
                    # create the new list of the less important RPs
                    no_pruning = ~(B_t_norm < gamma)

                # update matrices
                B_t  = B_t[no_pruning,:]
                D_x  = D_x[:,no_pruning]
                rp_X = rp_X[no_pruning,:]
                rp_y = rp_y[no_pruning,:]

            # e[t] = np.trace(E @ E.T) + self.lb * BB.sum()
        self.rp_X = rp_X
        self.rp_y = np.eye(self.y.shape[1])
        self.D_x  = D_x
        self.D_y  = D_y
        self.B    = B_t

    def fit_B(self): pass


# norm 2 regularization:
class L2_MLM(NN_MLM):
    def __init__(self, rp_number=None, C=1, random_state=42):
        # number of RPs and the regularization parameter
        self.C         = C
        super().__init__(rp_number=rp_number, random_state=42)
    def fit_B(self):
        # compute the distance regression matrix using OLS

        self.B = np.linalg.inv(self.D_x.T @ self.D_x + self.C * np.eye(self.rp_X.shape[0])) @ self.D_x.T @ self.D_y


class OS_MLMR(C_MLM):
    def __init__(self, norm=1, max_feature_number=None, feature_number=None, pinv=False,pp=True):
        self.norm               = norm
        self.feature_number     = feature_number
        self.max_feature_number = max_feature_number
        self.pinv               = pinv
        self.pp                 = pp

        if self.max_feature_number == None: self.max_feature_number = 0.20

        
    def select_RPs(self):
        # convert outputs to one-hot encoding
        # self.y = self.oh_convert(self.y)
        # compute pairwise distance matrices
        #  - D_x: input space
        #  - D_y: output space
        D_x  = cdist(self.X,self.X)
        
        rp_id     = np.random.choice(self.X.shape[0],3, replace=False)

        # self.D_y  = cdist(self.y, self.y[rp_id,:])
        self.D_y  = cdist(self.y, self.y)


        if self.max_feature_number <= 1: self.max_feature_number = int(self.max_feature_number * self.X.shape[0])
        if self.feature_number <= 1:     self.feature_number     = int(self.feature_number * self.X.shape[0])

        mrsr = MRSR(norm=self.norm,
                    max_feature_number=self.max_feature_number,
                    feature_number=self.feature_number,
                    pinv=self.pinv)

        mrsr.fit(D_x, self.D_y)

        self.X_rp_id = mrsr.order

        self.rp_X     = self.X[self.X_rp_id,:]
        self.rp_y     = self.y
        # self.rp_y     = self.y[rp_id,:]

        self.error = mrsr.error
        self.D_x = D_x[:,self.X_rp_id]
    
    def fit_B(self):
        self.B = pinv_(self.D_x) @ self.D_y
        # self.rp_y_pi = np.linalg.pinv(self.rp_y)
        # self.bias = (self.rp_y - self.__predict__(self.X)).mean()

    # def predict(self, X, y=None):
    #     errors.not_train(self)
    #     # compute matrix of distances from input RPs
    #     D_x = cdist(X,self.rp_X)
    #     # estimate matrix of distances from output RPs
    #     D_y_hat = D_x @ self.B
    #     ii = np.random.choice(self.rp_y.shape[0], 1)

    #     A = np.delete(2 * (self.rp_y - self.rp_y[ii,:]), ii, axis=0)
    #     b = np.delete(self.rp_y**2 + D_y_hat**2 - (self.rp_y[i,:]**2 - d_y_hat[i,:]**2) , i, axis=0)

    #     Y_hat = np.array([self.__predict__(X[i,:], D_y_hat[i,:][np.newaxis].T, ii) for i in range(X.shape[0])])
    #     return Y_hat

    # def __predict__(self, x, d_y_hat, i):
    #     A = np.delete(2 * (self.rp_y - self.rp_y[i,:]), i, axis=0)
    #     b = np.delete(self.rp_y**2 + d_y_hat**2 - (self.rp_y[i,:]**2 - d_y_hat[i,:]**2) , i, axis=0)
    #     y_hat = np.linalg.pinv(A) @ b
    #     return y_hat[0]



    # def __predict__(self, X, y=None):
    #     # compute matrix of distances from input RPs
    #     D_x = cdist(X,self.rp_X)
    #     # estimate matrix of distances from output RPs
    #     D_y_hat = D_x @ self.B

    #     y_hat = self.rp_y_pi @ D_y_hat.T
        
    #     return -y_hat.T


    # def predict(self, X):
    #     errors.not_train(self)
    #     return self.bias + self.__predict__(X)


    def plot(self,plt,X=None, y=None, figsize=None):
        # X = X if X != None else self.X
        # y = y if y != None else self.y

        X_ = np.linspace(X.min(), X.max(), 100)[np.newaxis].T
        y_ = self.predict(X_)

        if X.shape[1] == 1:
            fig = plt.figure(figsize=figsize) if figsize != None else plt.figure()
            plt.scatter(X,y, marker='o', c='orange')
            plt.scatter(X[self.X_rp_id,0],y[self.X_rp_id,0],alpha=0.9, facecolors='none',edgecolors='black',s=60,linewidths=2)
            y__ = gaussian_filter1d(y_, sigma=2)
            plt.plot(X_, y_, c='black')
        else:
            print("X have more that one dimensions.")



    # norm 2 regularization:
class ELM(BaseEstimator, RegressorMixin):
    def __init__(self, n_hidden=20, activation_func='sigmoid', random_state=42):
        self.n_hidden        = n_hidden
        self.activation_func = activation_func
        self.random_state    = random_state

    def oh_convert(self, y):
        if np.unique(y).shape[0] < 10:
            self.y_oh = False if len(y.shape) == 1 else True
            if self.y_oh == False: y = one_hot(y)
            return y
        else:
            return y


    def fit(self, X, y):
        self.X = np.concatenate((X, np.ones((X.shape[0],1))), axis=1)
        self.y = self.oh_convert(y)

        # Initialize the matrix B with values close to zero
        r = np.random.RandomState(self.random_state)
        self.W = r.normal(0,1, (self.X.shape[1], self.n_hidden))
        H = 1. / (1. + np.exp(-(self.X @ self.W)))
        self.M = np.linalg.pinv(H) @ self.y

    def predict(self, X, y=None):
        X = np.concatenate((X, np.ones((X.shape[0],1))), axis=1)
        H_hat = 1. / (1. + np.exp(-(X @ self.W)))
        y_hat = H_hat @ self.M

        if self.y_oh:
            return np.sign(y_hat)
        else:
            return y_hat.argmax(axis=1)

class ELMR(ELM):
    def predict(self, X, y=None):
        X = np.concatenate((X, np.ones((X.shape[0],1))), axis=1)
        H_hat = 1. / (1. + np.exp(-(X @ self.W)))
        return H_hat @ self.M

class OPELM(ELM):
    def fit(self, X, y):
        self.X = np.concatenate((X, np.ones((X.shape[0],1))), axis=1)
        self.y = self.oh_convert(y)

        # Initialize the matrix B with values close to zero
        r = np.random.RandomState(self.random_state)
        self.W = r.normal(0,1, (self.X.shape[1], self.n_hidden))
        H = 1. / (1. + np.exp(-(self.X @ self.W)))


        mrsr = MRSR(norm=1,max_feature_number=self.n_hidden-1)

        mrsr.fit(H, self.y)

        self.M = mrsr.W
        self.W = self.W[:,mrsr.order]

class OPELMR(OPELM):
    def predict(self, X, y=None):
        X = np.concatenate((X, np.ones((X.shape[0],1))), axis=1)
        H_hat = 1. / (1. + np.exp(-(X @ self.W)))
        return H_hat @ self.M
