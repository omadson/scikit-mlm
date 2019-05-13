# scikit-mlm
scikit-mlm is a Python module implementing the [Minimal Learning Machine][1] (MLM) machine learning technique using the [scikit-learn][2] API.

## quickstart
With NumPy, SciPy and scikit-learn available in your environment, install with:
```
pip install scikit-mlm
```

Classification example with the nearest neighbor MLM classifier:
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

## to-do list
### original proposal
 - [x] [regression (MLMR)](https://doi.org/10.1016/j.neucom.2014.11.073)
 - [x] [classification (MLMC)](https://doi.org/10.1016/j.neucom.2014.11.073)

### speed up
 - [x] [nearest neighbor MLM (NN_MLM)](https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec9)
 - [ ] [cubic equation MLM (C-MLM)](https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec10)

### reference points selection methods
#### classification
 - [x] [opposite neighborhood MLM (ON_MLM)](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2018-198.pdf)
 - [x] [fuzzy C-means MLM (FCM-MLM)](https://doi.org/10.1007/978-3-319-95312-0_34)
 - [ ] [&ell;<sub>1/2</sub>-norm regularization MLM (L12_MLM)](https://doi.org/10.1109/BRACIS.2018.00043)
 - [ ] [optimally selected MLM (OS-MLM)](https://doi.org/10.1007/978-3-030-03493-1_70)

<!-- #### regression
 - [ ] [regularized M-FOCUSS MLM (RMF_MLM)]() -->

### cost Sensitive
 - [x] [weighted MLM (w_MLM)](https://doi.org/10.1007/978-3-319-26532-2_61)

### missing values
 - [ ] [expected squared distance MLM (ESD-MLM)](https://doi.org/10.1007/978-3-319-26532-2_62)

### ensemble
 - [ ] [voting based MLM (V-MLM)](https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec11)
 - [ ] [weighted voting based MLM (WV-MLM)](https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec11)
 - [ ] [random sampling voting based MLM (RSV-MLM)](https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec11)
 - [ ] [random sampling weighted voting based MLM (RSWV-MLM)](https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec11)

### reject option
 - [ ] [reject option MLM (renjo-MLM)](https://doi.org/10.1109/BRACIS.2016.078)
 - [ ] [reject option weighted MLM (renjo-wMLM)](https://doi.org/10.1109/BRACIS.2016.078)

### ranking
 - [ ] [ranking MLM (R-MLM)](https://doi.org/10.1109/BRACIS.2015.39)

## how to cite scikit-mlm
if you use scikit-mlm in your paper, please cite
```
@misc{scikit-mlm,
    author       = "Madson Luiz Dantas Dias",
    year         = "2019",
    title        = "scikit-mlm: A implementation of {MLM} for scikit framework",
    url          = "https://github.com/omadson/scikit-mlm",
    institution  = "Federal University of Cear\'{a}, Department of Computer Science" 
}
```

## contributors
 - [Madson Dias](https://github.com/omadson)

## acknowledgement
 - thanks for [@JamesRitchie](https://github.com/JamesRitchie), the initial idea of this project is inspired on the [scikit-rvm](https://github.com/JamesRitchie/scikit-rvm) repo


[1]: https://doi.org/10.1016/j.neucom.2014.11.073
[2]: http://scikit-learn.org/
[3]: https://doi.org/10.1007/s11063-017-9587-5#
