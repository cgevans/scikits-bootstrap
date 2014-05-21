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

The package is licensed under the Modified BSD License.

Version Info
============

v0.3.2: This version contains various fixes to allow compatibility with Python
        3.3. While I have not used the package extensively with Python 3, all
        tests now pass, and importing works properly. The compatibility changes
        slightly modify the output of bootstrap_indexes, from a Python list to
        a Numpy array that can be iterated over in the same manner. This should
        only be important in extremely unusual situations.
