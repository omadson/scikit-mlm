# scikit-mlm\*
scikit-mlm is a Python module implementing the [Minimal Learning Machine][1] (MLM) machine learning technique using the [scikit-learn][2] API.

## Quickstart
With NumPy, SciPy and scikit-learn available in your environment, install with:
```
pip3 install https://github.com/omadson/scikit-mlm/archive/master.zip
```

Regression example with the MLMR class:
```Python
import numpy as np
from skmlm import MLMR
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


boston = load_boston()

X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)

clf = MLMR(rp_number=130)
clf.fit(X_train, y_train)

y_hat = clf.predict(X_test)

mean_squared_error(y_test, y_hat)
```

## To-do list
 - [x] [original proposal for regression](https://doi.org/10.1016/j.neucom.2014.11.073)
 - [ ] [original proposal for classification](https://doi.org/10.1016/j.neucom.2014.11.073)

### speed up
 - [ ] [nearest neighbor MLM (NN-MLM)](https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec9)
 - [ ] [cubic equation MLM (C-MLM)](https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec10)


### Ensemble
 - [ ] [voting based MLM (V-MLM)](https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec11)
 - [ ] [weighted voting based MLM (WV-MLM)](https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec11)
 - [ ] [random sampling voting based MLM (RSV-MLM)](https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec11)
 - [ ] [random sampling weighted voting based MLM (RSWV-MLM)](https://link.springer.com/article/10.1007%2Fs11063-017-9587-5#Sec11)

### Missing values
 - [ ] [expected squared distance MLM (ESD-MLM)](https://doi.org/10.1007/978-3-319-26532-2_62)

### Reject option
 - [ ] [Efficient minimal learning machines with reject option](https://doi.org/10.1109/BRACIS.2016.078)

### Reference points selection methods
 - [ ] [opposite neighborhood MLM (ON-MLM)](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2018-198.pdf)
 - [ ] [A Fuzzy C-means MLM (FCM-MLM)](https://doi.org/10.1007/978-3-319-95312-0_34)

<!-- ### unclassified
 - [ ] [MLM-rank: A Ranking Algorithm Based on the Minimal Learning Machine](https://doi.org/10.1109/BRACIS.2015.39) -->

## Contributors
 - [Madson Dias](https://github.com/omadson)

---

\* the initial idea of this project is based on the [scikit-rvm](https://github.com/JamesRitchie/scikit-rvm) repo, of  [@JamesRitchie](https://github.com/JamesRitchie). The contributors acknowledge your help :heart:.


[1]: https://doi.org/10.1016/j.neucom.2014.11.073
[2]: http://scikit-learn.org/
[3]: https://doi.org/10.1007/s11063-017-9587-5#
