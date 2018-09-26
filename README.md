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
## Contributors
 - [Madson Dias](https://github.com/omadson)

---

\* the initial idea of this project is based on the [scikit-rvm](https://github.com/JamesRitchie/scikit-rvm) repo, of  @JamesRitchie. The contributors acknowledge your help :heart:.


[1]: https://doi.org/10.1016/j.neucom.2014.11.073
[2]: http://scikit-learn.org/
