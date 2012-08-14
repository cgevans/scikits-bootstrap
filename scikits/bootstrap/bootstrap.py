from numpy.random import randint
from scipy.stats import norm
import numpy as np
import warnings

class InstabilityWarning(UserWarning):
    """Issued when results may be unstable."""
    pass

# On import, make sure that InstabilityWarnings are not filtered out.
warnings.simplefilter('always',InstabilityWarning)

def ci(data, statfunction=np.mean, alpha=0.05, n_samples=10000, method='bca'):
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
alpha: float or iterable, optional
    The percentiles to use for the confidence interval (default=0.05). If this
    is a float, the returned values are (alpha/2, 1-alpha/2) percentile confidence
    intervals. If it is an iterable, alpha is assumed to be an iterable of
    each desired percentile.
n_samples: float, optional
    The number of bootstrap samples to use (default=10000)
method: string
    The method to use: one of 'pi' or 'bca' (default='bca')

Returns
-------
confidences: tuple of floats
    The confidence percentiles specified by alpha

Calculation Methods
-------------------
'pi': Percentile Interval (Efron 13.3)
    The percentile interval method simply returns the 100*alphath bootstrap
    sample's values for the statistic.

    This is an extremely simple method of confidence interval calculation.
    However, it has several disadvantages compared to the bias-corrected
    accelerated method, which is the default.
'bca': Bias-Corrected Accelerated Non-Parametric (Efron 14.3) (default)
    This method is much more complex to explain. However, it gives considerably
    better results, and is generally recommended for normal situations. Note
    that in cases where the statistic is smooth, and can be expressed with
    weights, the ABC method will give approximated results much, much faster.

References
----------
Efron, An Introduction to the Bootstrap. Chapman & Hall 1993
    """

    # Deal with the alpha values
    if np.iterable(alpha):
        alphas = np.array(alpha)
    else:
        alphas = np.array([alpha/2,1-alpha/2])

    # Ensure that the data is actually an array. This isn't nice to pandas,
    # but pandas seems much much slower and the indexes become a problem.
    data = np.array(data)

    # We don't need to generate actual samples; that would take more memory.
    # Instead, we can generate just the indexes, and then apply the statfun
    # to those indexes.
    bootindexes = bootstrap_indexes( data, n_samples )
    stat = np.array([statfunction(data[indexes]) for indexes in bootindexes])
    stat.sort()

    # Percentile Interval Method
    if method == 'pi':
        avals = alphas

    # Bias-Corrected Accelerated Method
    elif method == 'bca':

        # The value of the statistic function applied just to the actual data.
        ostat = statfunction(data)

        # The bias correction value.
        z0 = norm.ppf( ( 1.0*np.sum(stat < ostat, axis=0)  ) / n_samples )

        # Statistics of the jackknife distribution
        jackindexes = jackknife_indexes(data)
        jstat = [statfunction(data[index]) for index in jackindexes]
        jmean = np.mean(jstat,axis=0)

        # Acceleration value
        a = np.sum( (jmean - jstat)**3, axis=0 ) / ( 6.0 * np.sum( (jmean - jstat)**2, axis=0)**1.5 )

        zs = z0 + norm.ppf(alphas)

        avals = norm.cdf(z0 + zs/(1-a*zs))

    else:
        raise ValueError("Method {0} is not supported.".format(method))

    nvals = np.round((n_samples-1)*avals).astype('int')

    if np.any(nvals==0) or np.any(nvals==n_samples-1):
        warnings.warn("Some values used extremal samples; results are probably unstable.", InstabilityWarning)
    elif np.any(nvals<10) or np.any(nvals>=n_samples-10):
        warnings.warn("Some values used top 10 low/high samples; results may be unstable.", InstabilityWarning)

    return stat[np.round((n_samples-1)*avals).astype('int')]



def ci_abc(data, stat=lambda x,y: np.average(x,weights=y), alpha=0.05, epsilon = 0.001):
    """
Given a set of data ``data``, and a statistics function ``statfunction`` that
applies to that data, computes the non-parametric approximate bootstrap
confidence (ABC) interval for ``stat`` on that data. Data points are assumed
to be delineated by axis 0.

Parameters
----------
data: array_like, shape (N, ...)
    Input data. Data points are assumed to be delineated by axis 0. Beyond this,
    the shape doesn't matter, so long as ``statfunction`` can be applied to the
    array.
stat: function (data, weights) -> value
    The _weighted_ statistic function. This must accept weights, unlike for other
    methods.
alpha: float or iterable, optional
    The percentiles to use for the confidence interval (default=0.05). If this
    is a float, the returned values are (alpha/2, 1-alpha/2) percentile confidence
    intervals. If it is an iterable, alpha is assumed to be an iterable of
    each desired percentile.
epsilon: float
    The step size for finite difference calculations. (default=0.001)

Returns
-------
confidences: tuple of floats
    The confidence percentiles specified by alpha

References
----------
Efron, An Introduction to the Bootstrap. Chapman & Hall 1993
bootstrap R package: http://cran.r-project.org/web/packages/bootstrap/
    """
    # Deal with the alpha values
    if not np.iterable(alpha):
        alpha = np.array([alpha/2,1-alpha/2])
    else:
        alpha = np.array(alpha)

    # Ensure that the data is actually an array. This isn't nice to pandas,
    # but pandas seems much much slower and the indexes become a problem.
    data = np.array(data)

    n = data.shape[0]*1.0
    nn = data.shape[0]

    I = np.identity(nn)
    ep = epsilon / n*1.0
    p0 = np.repeat(1.0/n,nn)

    t1 = np.zeros(nn); t2 = np.zeros(nn)
    t0 = stat(data,p0)

    # There MUST be a better way to do this!
    for i in range(0,nn):
        di = I[i] - p0
        tp = stat(data,p0+ep*di)
        tm = stat(data,p0-ep*di)
        t1[i] = (tp-tm)/(2*ep)
        t2[i] = (tp-2*t0+tm)/ep**2

    sighat = np.sqrt(np.sum(t1**2))/n
    a = (np.sum(t1**3))/(6*n**3*sighat**3)
    delta = t1/(n**2*sighat)
    cq = (stat(data,p0+ep*delta)-2*t0+stat(data,p0-ep*delta))/(2*sighat*ep**2)
    bhat = np.sum(t2)/(2*n**2)
    curv = bhat/sighat-cq
    z0 = norm.ppf(2*norm.cdf(a)*norm.cdf(-curv))
    Z = z0+norm.ppf(alpha)
    za = Z/(1-a*Z)**2
    # stan = t0 + sighat * norm.ppf(alpha)
    abc = np.zeros_like(alpha)
    for i in range(0,len(alpha)):
        abc[i] = stat(data,p0+za[i]*delta)

    return abc

def bootstrap_indexes(data, n_samples=10000):
    """
Given data points data, where axis 0 is considered to delineate points, return
a list of arrays where each array is a set of bootstrap indexes.
    """
    return ( randint(data.shape[0],size=data.shape[0] ) for a in xrange(0,n_samples) )

def jackknife_indexes(data):
    """
Given data points data, where axis 0 is considered to delineate points, return
a list of arrays where each array is a set of jackknife indexes.

For a given set of data Y, the jackknife sample J[i] is defined as the data set
Y with the ith data point deleted.
    """
    base = np.arange(0,len(data))
    return (np.delete(base,i) for i in base)

def subsample_indexes(data, n_samples=1000, size=0.5):
    """
Given data points data, where axis 0 is considered to delineate points, return
a list of arrays where each array is indexes a subsample of the data of size
``size``. If size is >= 1, then it will be taken to be an absolute size. If
size < 1, it will be taken to be a fraction of the data size. If size == -1, it
will be taken to mean subsamples the same size as the sample (ie, permuted
samples)
    """
    if size == -1:
        size = len(data)
    elif (size < 1) and (size > 0):
        size = round(size*len(data))
    elif size > 1:
        pass
    else:
        raise ValueError("size cannot be {0}".format(size))
    base = np.tile(np.arange(len(data)),(n_samples,1))
    for sample in base: np.random.shuffle(sample)
    return base[:,0:size]

