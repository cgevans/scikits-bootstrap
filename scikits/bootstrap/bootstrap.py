from numpy.random import randint
from scipy.stats import norm
import numpy as np

def ci(data, statfunction, alpha=0.05, n_samples=10000, method='bca'):
    """
Given a set of data ``data``, and a statistics function ``statfunction`` that
applies to that data, computes the bootstrap confidence interval for
``statfunction`` on that data. Data points are assumed to be delineated by
axis 0.

Parameters
----------
data: array_like, shape (N, ...)
    Input data. Data points are assumed to be delineated by axis 0. Beyond this,
    the shape doesn't matter, so long as ``statfunction`` can be applied to the
    array.
statfunction: function
    This function should accept samples of data from ``data``. It is applied
    to these samples individually.
alpha: float, optional
    The percentile to use for the confidence interval (default=0.05)
n_samples: float, optional
    The number of bootstrap samples to use (default=10000)
method: string
    The method to use: one of 'pi' or 'bca' (default='bca')

Returns
-------
(lower, upper): tuple of floats

Calculation Methods
-------------------
'pi': Percentile Interval
    The percentile interval method simply returns the 100*alpha and
    100*(1-alpha)th bootstrap sample's values for the statistic. See
    Efron, section 13.3.

    This is an extremely simple method of confidence interval calculation.
    However, it has several disadvantages compared to the bias-corrected
    accelerated method, which is the default.
'bca': Bias-Corrected Accelerated Non-Parametric (Efron 14.3) (default)
    This method is much more complex to explain. However, ... (FIXME)
'abc': Approximate Bootstrap Confidence (Efron 14.4)
    This method is not yet implemented.

References
----------
Efron, An Introduction to the Bootstrap. Chapman & Hall 1993
    """

    # We don't need to generate actual samples; that would take more memory.
    # Instead, we can generate just the indexes, and then apply the statfun
    # to those indexes.
    bootindexes = bootstrap_indexes( data, n_samples )
    stat = np.array([statfunction(data[indexes]) for indexes in bootindexes])
    stat_sorted = np.sort(stat)

    # Percentile Interval Method
    if method == 'pi':
        return ( stat_sorted[round(n_samples*alpha/2)], stat_sorted[round(n_samples*(1-alpha/2))] )

    # Bias-Corrected Accelerated Method
    elif method == 'bca':

        # The value of the statistic function applied just to the actual data.
        ostat = statfunction(data)

        # The bias correction value.
        z0 = norm.ppf( ( 1.0*np.sum(stat < ostat)  ) / n_samples )

        # Statistics of the jackknife distribution
        jackindexes = jackknife_indexes(data)
        jstat = [statfunction(x) for x in data[jackindexes]]
        jmean = np.mean(jstat)

        # Acceleration value
        a = np.sum( (jstat - jmean)**3 ) / ( 6.0 * np.sum( (jstat - jmean)**2 )**1.5 )
        
        zp = z0 + norm.ppf(1-alpha/2)
        zm = z0 - norm.ppf(1-alpha/2)

        a1 = norm.cdf(z0 + zm/(1-a*zm))
        a2 = norm.cdf(z0 + zp/(1-a*zp))

        return (stat_sorted[np.round(n_samples*a1)],stat_sorted[np.round(n_samples*a2)])
        
    else:
        raise ValueError()
        

def bootstrap_indexes(data, n_samples=10000):
    """
Given data points data, where axis 0 is considered to delineate points, return
a list of arrays where each array is a set of bootstrap indexes.
    """
    return randint(data.shape[0],size=(n_samples,data.shape[0]))

def jackknife_indexes(data):
    """
Given data points data, where axis 0 is considered to delineate points, return
a list of arrays where each array is a set of jackknife indexes.

For a given set of data Y, the jackknife sample J[i] is defined as the data set
Y with the ith data point deleted.
    """
    # FIXME: this is rather messy.
    return (lambda n: np.delete(np.tile(np.array(range(0,n)),n),range(0,n*n,n+1)).reshape((n,n-1)))(len(data))

