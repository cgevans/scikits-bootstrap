[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3548989.svg)](https://doi.org/10.5281/zenodo.3548989)
[![Codecov](https://img.shields.io/codecov/c/github/cgevans/scikits-bootstrap)](https://codecov.io/gh/cgevans/scikits-bootstrap)
[![PyPI](https://img.shields.io/pypi/v/scikits-bootstrap)](https://pypi.org/project/scikits.bootstrap/)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/cgevans/scikits-bootstrap)]
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/scikits-bootstrap)

scikits-bootstrap
=================

Scikits.bootstrap provides bootstrap confidence interval algorithms for
Numpy/Scipy/Pandas. It originally required scipy, but no longer needs
it.

It also provides an algorithm which estimates the probability that the
statistics lies satisfies some criteria, e.g. lies in some interval.

At present, it is rather feature-incomplete and in flux. However, the
functions that have been written should be relatively stable as far as
results.

Much of the code has been written based off the descriptions from Efron
and Tibshirani's Introduction to the Bootstrap, and results should match
the results obtained from following those explanations. However, the
current ABC code is based off of the modified-BSD-licensed R port of the
Efron bootstrap code, as I do not believe I currently have a sufficient
understanding of the ABC method to write the code independently.

In any case, please contact me (Constantine Evans
<cevans@evanslabs.org>) with any questions or suggestions. I'm trying to
add documentation, and will be adding tests as well. I'm especially
interested, however, in how the API should actually look; please let me
know if you think the package should be organized differently.

The package is licensed under the BSD 3-Clause License. It is supported
in part by the Evans Foundation.

Version Info
============

-   v1.1.0-pre.1: Randomness is now generated via a numpy.random
    Generator. Anything that relied on using numpy.random.seed to obtain
    deterministic results will fail (mostly of relevance for testing).
    Seeds (or Generators) can now be passed to relevant functions with
    the `seed` argument, but note that changes in Numpy's random number
    generation means this will not give the same results that would be
    obtained using `numpy.random.seed` to set the seed in previous
    versions.

    There is a new pval function, and there are several bugfixes.

    Numba is now supported in some instances (np.average or np.mean as
    statfunction, 1-D data), using use\_numba=True. Pypy3 is also
    supported. Typing information has been added.

    Handling of multiple data sets (tuples/etc of arrays) now can be
    specified as multi="paired" (the previous handling), where the sets
    must be of the same length, and samples are taken keeping
    corresponding points connected, or multi="independent", treating
    data sets as independent and sampling them seperately (in which case
    they may be different sizes).

-   v1.0.1: Licensing information added.

-   v1.0.0: scikits.bootstrap now uses pyerf, which means that it
    doesn't actually need scipy at all. It should work with PyPy, has
    some improved error and warning messages, and should be a bit faster
    in many cases. The old ci\_abc function has been removed: use
    method='abc' instead.

-   v0.3.3: Bug fixes. Warnings have been cleaned up, and are
    implemented for BCa when all statistic values are equal (a common
    confusion in prior versions). Related numpy warnings are now
    suppressed. Some tests on Python 2 were fixed, and the PyPI website
    link is now correct.

-   v0.3.2: This version contains various fixes to allow compatibility
    with Python 3.3. While I have not used the package extensively with
    Python 3, all tests now pass, and importing works properly. The
    compatibility changes slightly modify the output of
    bootstrap\_indexes, from a Python list to a Numpy array that can be
    iterated over in the same manner. This should only be important in
    extremely unusual situations.

Installation and Usage
======================

scikits.bootstrap is tested on Python 3.6 - 3.9, and PyPy 3. The package
can be installed using pip.

`pip install scikits.bootstrap`

Usage example for python 3.x:

    import scikits.bootstrap as boot
    import numpy as np
    boot.ci(np.random.rand(100), np.average)
