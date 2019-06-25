import itertools
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from sklearn.utils import shuffle

# UTILS functions used in some codes

# def compute_reduction

# red = lambda a: a.named_steps['gridsearchcv'].best_estimator_.B.shape[0] / a.named_steps['gridsearchcv'].best_estimator_.X.shape[0]
def get_metrics(dataset_name, model_name, scores):
    param_keys = list(scores['estimator'][0].named_steps['gridsearchcv'].best_estimator_.get_params().keys())
    params = dict()
    for param_key in param_keys: params[param_key] = list()
    best_params = [scores['estimator'][i].named_steps['gridsearchcv'].best_estimator_.get_params() for i in range(len(scores['estimator']))]
    for best_param in best_params:
        for parameter in best_param.keys():
            params[parameter].append(best_param[parameter])
    params['fit_time']      = scores['fit_time']
    params['score_time']    = scores['score_time']
    params['test_accuracy'] = scores['test_accuracy']

    params['best_estimator'] = [scores['estimator'][i].named_steps['gridsearchcv'].best_estimator_ for i in range(len(scores['estimator']))]

    header_ = list(params.keys())

    header  = [i for i in itertools.product([dataset_name],[model_name],header_)]
    ordered_header = [k for i,j,k in header]
    return pd.DataFrame(pd.DataFrame(params)[ordered_header].values, columns=pd.MultiIndex.from_tuples(header))

def get_metrics_MLM_gs(dataset_name, model_name, scores):
    params = dict()
    e_name = list(filter(lambda x: x in ['opelm','os_mlm', 'nn_mlm', 'gridsearchcv'], scores['estimator'][0].named_steps.keys()))[0]
    if e_name == 'gridsearchcv':
        param_keys = list(scores['estimator'][0].named_steps['gridsearchcv'].best_estimator_.get_params().keys())
        params['best_estimator'] = [scores['estimator'][i].named_steps['gridsearchcv'].best_estimator_ for i in range(len(scores['estimator']))]
        params['irp_number'] = [e.B.shape[0] for e in params['best_estimator']]
    elif e_name in ['os_mlm','nn_mlm']:
        param_keys = list(scores['estimator'][0].named_steps[e_name].get_params().keys())
        params['best_estimator'] = [scores['estimator'][i].named_steps[e_name] for i in range(len(scores['estimator']))]
        params['irp_number'] = [e.B.shape[0] for e in params['best_estimator']]
    elif e_name == 'opelm':
        param_keys = list(scores['estimator'][0].named_steps[e_name].get_params().keys())
        params['best_estimator'] = [scores['estimator'][i].named_steps[e_name] for i in range(len(scores['estimator']))]
        params['n_hidden_f'] = [e.W.shape[1] for e in params['best_estimator']]

    for param_key in param_keys: params[param_key] = list()
    best_params = [e.get_params() for e in params['best_estimator']]
    

    for best_param in best_params:
        for parameter in best_param.keys():
            params[parameter].append(best_param[parameter])
    

    
    

    params['fit_time']      = scores['fit_time']
    params['score_time']    = scores['score_time']
    params['test_accuracy'] = scores['test_accuracy']

    header_ = list(params.keys())
    header  = [i for i in itertools.product([dataset_name],[model_name],header_)]
    ordered_header = [k for i,j,k in header]
    return pd.DataFrame(pd.DataFrame(params)[ordered_header].values, columns=pd.MultiIndex.from_tuples(header))

def load_dataset(url=None, task='classification',type="artificial", name="two_squares", perm=True, random_state=42):
    url = 'https://raw.githubusercontent.com/omadson/datasets/master' if url == None else url
    data = pd.read_csv('%s/%s/%s/%s/data.csv' % (url, task, type, name), header=None)
    dataset      = dict()
    if perm:
        return shuffle(data.iloc[:,:-1].values,data.iloc[:,-1].values, random_state=random_state)
    else:
        return data.iloc[:,:-1].values,data.iloc[:,-1].values

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
