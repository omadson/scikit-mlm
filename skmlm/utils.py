import numpy as np
from scipy.spatial.distance import cdist
# UTILS functions used in some codes

# one-hot: convert output to one-hot encoding
def one_hot(y):    
    y = [int(i) for i in y.tolist()]
    l = len(y)
    c = len(np.unique(y))
    y_oh = np.zeros((l, c))
    y_oh[np.arange(l), y] = 1
    return y_oh


# opposite neighborhood
def ON(X, y, neighborhood_size=1, D_in=None, D_out=None):
    if neighborhood_size <= X.shape[0]:
        D_in  = D_in  if isinstance(D_in, np.ndarray)  else cdist(X,X)
        D_out = D_out if isinstance(D_out, np.ndarray) else cdist(y,y)

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