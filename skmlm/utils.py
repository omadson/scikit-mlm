import numpy as np
from scipy.spatial.distance import cdist
# UTILS functions used in some codes

# def compute_reduction

# red = lambda a: a.named_steps['gridsearchcv'].best_estimator_.B.shape[0] / a.named_steps['gridsearchcv'].best_estimator_.X.shape[0]


def pinv_(X):
    try:
        return np.linalg.inv(X.T @ X) @ X.T
    except Exception as e:
        return np.linalg.pinv(X)


# one-hot: convert output to one-hot encoding
def one_hot(y):
    y = [int(i) for i in y.tolist()]
    l = len(y)
    c = len(np.unique(y))
    y_oh = np.zeros((l, c))
    y_oh[np.arange(l), y] = 1
    return y_oh


# opposite neighborhood
def ON(X, y, neighborhood_size=1, D_x=None, D_y=None):
    # verify if <neighborhood_size> is small of the small class-samples
    if neighborhood_size <= X.shape[0]:
        # compute distance matrices
        D_x  = D_x  if isinstance(D_x, np.ndarray)  else cdist(X,X)
        D_y  = D_y  if isinstance(D_y, np.ndarray)  else cdist(y,y)

        # transform y into ordered
        if len(y.shape) != 1:
            yy = y.argmax(axis=1)
        on = set()
        for i in range(X.shape[0]):
            # sort the distance from sample x_i
            d_ord    = D_x[i,:].argsort()
            # select samples of other classes
            opp = yy[d_ord] != yy[i]
            # create set of opposite neighborhood from sample x_i
            on_i = set(d_ord[opp][:neighborhood_size])
            # add to the opposite neighborhood set
            on = on.union(on_i)

        on_index = np.zeros(X.shape[0])
        on_index[list(on)] = 1
        return on_index == 1, D_x, D_y

class ERRORS():
    def not_train(self, obj):
        try:
                getattr(obj, "B")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
