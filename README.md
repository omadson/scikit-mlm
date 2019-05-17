# scikit-mlm

![GitHub](https://img.shields.io/github/license/omadson/scikit-mlm.svg)
[![PyPI](https://img.shields.io/pypi/v/scikit-mlm.svg)](http://pypi.org/project/scikit-mlm/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/w/omadson/scikit-mlm.svg)](https://github.com/omadson/scikit-mlm/pulse)
[![GitHub last commit](https://img.shields.io/github/last-commit/omadson/scikit-mlm.svg)](https://github.com/omadson/scikit-mlm/commit/master)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2822613.svg)](https://doi.org/10.5281/zenodo.2822613)

`scikit-mlm` is a Python module implementing the [Minimal Learning Machine][1] (MLM) machine learning technique using the [scikit-learn][2] API.

## instalation
the `scikit-mlm` package is available in [PyPI](https://pypi.org/project/scikit-mlm/). to install, simply type the following command:
```
pip install scikit-mlm
```

## basic usage
example of classification with the [nearest neighbor MLM](https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec9) classifier:
```Python
from skmlm import NN_MLM
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris

# load dataset
dataset = load_iris()

clf = make_pipeline(MinMaxScaler(), NN_MLM(rp_number=20))
scores = cross_val_score(clf, dataset.data, dataset.target, cv=10, scoring='accuracy')

print('AVG = %.3f, STD = %.3f' % (scores.mean(), scores.std()))
```

## how to cite scikit-mlm
if you use `scikit-mlm` in your paper, please cite it in your publication.
```
@misc{scikit-mlm,
    author       = "Madson Luiz Dantas Dias",
    year         = "2019",
    title        = "scikit-mlm: An implementation of {MLM} for scikit-learn framework",
    url          = "https://github.com/omadson/scikit-mlm",
    doi          = "10.5281/zenodo.2822613",
    institution  = "Federal University of Cear\'{a}, Department of Computer Science" 
}
```

## contributing

this project is open for contributions. here are some of the ways for you to contribute:
 - bug reports/fix
 - features requests
 - use-case demonstrations

to make a contribution, just fork this repository, push the changes in your fork, open up an issue, and make a pull request!

## list of implemented technics
 - [original regression (MLMR)](https://doi.org/10.1016/j.neucom.2014.11.073)
 - [original classification (MLMC)](https://doi.org/10.1016/j.neucom.2014.11.073)
 - [nearest neighbor MLM (NN_MLM)](https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec9)
 - [opposite neighborhood MLM (ON_MLM)](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2018-198.pdf)
 - [fuzzy C-means MLM (FCM_MLM)](https://doi.org/10.1007/978-3-319-95312-0_34)
 - [optimally selected MLM (OS_MLM)](https://doi.org/10.1007/978-3-030-03493-1_70)
 - [&ell;<sub>1/2</sub>-norm regularization MLM (L12_MLM)](https://doi.org/10.1109/BRACIS.2018.00043)
 - [weighted MLM (w_MLM)](https://doi.org/10.1007/978-3-319-26532-2_61)



## future improvements

list of methods that will be implemented in the next releases:
 - [cubic equation MLM (C-MLM)](https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec10)
 - [expected squared distance MLM (ESD-MLM)](https://doi.org/10.1007/978-3-319-26532-2_62)
 - [voting based MLM (V-MLM)](https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec11)
 - [weighted voting based MLM (WV-MLM)](https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec11)
 - [random sampling voting based MLM (RSV-MLM)](https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec11)
 - [random sampling weighted voting based MLM (RSWV-MLM)](https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec11)
 - [reject option MLM (renjo-MLM)](https://doi.org/10.1109/BRACIS.2016.078)
 - [reject option weighted MLM (renjo-wMLM)](https://doi.org/10.1109/BRACIS.2016.078)
 - [ranking MLM (R-MLM)](https://doi.org/10.1109/BRACIS.2015.39)

<!-- #### regression
 - [ ] [regularized M-FOCUSS MLM (RMF_MLM)]() -->

<!-- ### speed up
### missing values
### ensemble 
### reject option
### ranking -->

## contributors
 - [Madson Dias](https://github.com/omadson)

## acknowledgement
 - thanks for [@JamesRitchie](https://github.com/JamesRitchie), the initial idea of this project is inspired on the [scikit-rvm](https://github.com/JamesRitchie/scikit-rvm) repo


[1]: https://doi.org/10.1016/j.neucom.2014.11.073
[2]: http://scikit-learn.org/
[3]: https://doi.org/10.1007/s11063-017-9587-5#
