scikits-bootstrap
=================

Scikits.bootstrap provides bootstrap confidence interval algorithms for scipy.

It also provides an algorithm which estimates the probability that the statistics
lies satisfies some criteria, e.g. lies in some interval.

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

