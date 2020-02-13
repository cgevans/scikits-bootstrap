[![Travis](https://travis-ci.org/cgevans/scikits-bootstrap.svg?branch=master)](https://travis-ci.org/cgevans/scikits-bootstrap)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3548989.svg)](https://doi.org/10.5281/zenodo.3548989)

scikits-bootstrap
=================

Scikits.bootstrap provides bootstrap confidence interval algorithms for scipy.

At present, it is rather feature-incomplete and in flux. However, the functions
that have been written should be relatively stable as far as results.

Much of the code has been written based off the descriptions from Efron and
Tibshirani's Introduction to the Bootstrap, and results should match the results
obtained from following those explanations. However, the current ABC code is
based off of the modified-BSD-licensed R port of the Efron bootstrap code, as
I do not believe I currently have a sufficient understanding of the ABC method
to write the code independently.

In any case, please contact me (Constantine Evans <cevans@evanslabs.org>) with
any questions or suggestions. I'm trying to add documentation, and will
be adding tests as well. I'm especially interested, however, in how the API
should actually look; please let me know if you think the package should be
organized differently.

The package is licensed under the Modified BSD License. It is supported in part
by the Evans Foundation.

Version Info
============

v1.0.1: Licensing information added.

v1.0.0: scikits.bootstrap now uses pyerf, which means that it doesn't actually
        need scipy at all.  It should work with PyPy, has some improved error
		and warning messages, and should be a bit faster in many cases.  The old
		ci_abc function has been removed: use method='abc' instead.

v0.3.3: Bug fixes.  Warnings have been cleaned up, and are implemented for BCa
        when all statistic values are equal (a common confusion in prior versions).
		Related numpy warnings are now suppressed.  Some tests on Python 2 were
		fixed, and the PyPI website link is now correct.

v0.3.2: This version contains various fixes to allow compatibility with Python
        3.3. While I have not used the package extensively with Python 3, all
        tests now pass, and importing works properly. The compatibility changes
        slightly modify the output of bootstrap_indexes, from a Python list to
        a Numpy array that can be iterated over in the same manner. This should
        only be important in extremely unusual situations.



Installation and Usage
======================
As described (<http://scikits.appspot.com/bootstrap>), the package can be installed using pip.

`pip install scikits.bootstrap`

Usage example for python 3.x:

```
import scikits.bootstrap as boot
import numpy as np
boot.ci(np.random.rand(100), np.average)
```
