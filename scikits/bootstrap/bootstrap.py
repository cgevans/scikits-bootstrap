from __future__ import absolute_import, division, print_function

from numpy.random import randint
from math import ceil, sqrt
import numpy as np
import pyerf
import warnings

s2 = sqrt(2)


def _ncdf_py(x):
    return 0.5*(1+pyerf.erf(x/s2))


def _nppf_py(x):
    return s2 * pyerf.erfinv(2*x-1)


nppf = np.vectorize(_nppf_py, [np.float])
ncdf = np.vectorize(_ncdf_py, [np.float])


__version__ = '1.0.0'

# Keep python 2/3 compatibility, without using six. At some point,
# we may need to add six as a requirement, but right now we can avoid it.
try:
    xrange
except NameError:
    xrange = range


class InstabilityWarning(UserWarning):
    """Issued when results may be unstable."""
    pass


# On import, make sure that InstabilityWarnings are not filtered out.
warnings.simplefilter('always', InstabilityWarning)


def ci(data, statfunction=None, alpha=0.05, n_samples=10000,
       method='bca', output='lowhigh', epsilon=0.001, multi=None,
       _iter=True):
    """
Given a set of data ``data``, and a statistics function ``statfunction`` that
applies to that data, computes the bootstrap confidence interval for
``statfunction`` on that data. Data points are assumed to be delineated by
axis 0.

Parameters
----------
data: array_like, shape (N, ...) OR tuple of array_like all with shape (N, ...)
    Input data. Data points are assumed to be delineated by axis 0. Beyond this,
    the shape doesn't matter, so long as ``statfunction`` can be applied to the
    array. If a tuple of array_likes is passed, then samples from each array (along
    axis 0) are passed in order as separate parameters to the statfunction. The
    type of data (single array or tuple of arrays) can be explicitly specified
    by the multi parameter.
statfunction: function (data, weights=(weights, optional)) -> value
    This function should accept samples of data from ``data``. It is applied
    to these samples individually. 
    
    If using the ABC method, the function _must_ accept a named ``weights`` 
    parameter which will be an array_like with weights for each sample, and 
    must return a _weighted_ result. Otherwise this parameter is not used
    or required. Note that numpy's np.average accepts this. (default=np.average)
alpha: float or iterable, optional
    The percentiles to use for the confidence interval (default=0.05). If this
    is a float, the returned values are (alpha/2, 1-alpha/2) percentile confidence
    intervals. If it is an iterable, alpha is assumed to be an iterable of
    each desired percentile.
n_samples: float, optional
    The number of bootstrap samples to use (default=10000)
method: string, optional
    The method to use: one of 'pi', 'bca', or 'abc' (default='bca')
output: string, optional
    The format of the output. 'lowhigh' gives low and high confidence interval
    values. 'errorbar' gives transposed abs(value-confidence interval value) values
    that are suitable for use with matplotlib's errorbar function. (default='lowhigh')
epsilon: float, optional (only for ABC method)
    The step size for finite difference calculations in the ABC method. Ignored for
    all other methods. (default=0.001)
multi: boolean, optional
    If False, assume data is a single array. If True, assume data is a tuple/other
    iterable of arrays of the same length that should be sampled together. If None,
    decide based on whether the data is an actual tuple. (default=None)
    
Returns
-------
confidences: tuple of floats
    The confidence percentiles specified by alpha

Calculation Methods
-------------------
'pi': Percentile Interval (Efron 13.3)
    The percentile interval method simply returns the 100*alphath bootstrap
    sample's values for the statistic. This is an extremely simple method of 
    confidence interval calculation. However, it has several disadvantages 
    compared to the bias-corrected accelerated method, which is the default.
'bca': Bias-Corrected Accelerated (BCa) Non-Parametric (Efron 14.3) (default)
    This method is much more complex to explain. However, it gives considerably
    better results, and is generally recommended for normal situations. Note
    that in cases where the statistic is smooth, and can be expressed with
    weights, the ABC method will give approximated results much, much faster.
    Note that in a case where the statfunction results in equal output for every
    bootstrap sample, the BCa confidence interval is technically undefined, as
    the acceleration value is undefined. To match the percentile interval method
    and give reasonable output, the implementation of this method returns a
    confidence interval of zero width using the 0th bootstrap sample in this
    case, and warns the user.  
'abc': Approximate Bootstrap Confidence (Efron 14.4, 22.6)
    This method provides approximated bootstrap confidence intervals without
    actually taking bootstrap samples. This requires that the statistic be 
    smooth, and allow for weighting of individual points with a weights=
    parameter (note that np.average allows this). This is _much_ faster
    than all other methods for situations where it can be used.

Examples
--------
To calculate the confidence intervals for the mean of some numbers:

>> boot.ci( np.randn(100), np.average )

Given some data points in arrays x and y calculate the confidence intervals
for all linear regression coefficients simultaneously:

>> boot.ci( (x,y), scipy.stats.linregress )

References
----------
Efron, An Introduction to the Bootstrap. Chapman & Hall 1993
    """

    # Deal with the alpha values
    if np.iterable(alpha):
        alphas = np.array(alpha)
    else:
        alphas = np.array([alpha/2, 1-alpha/2])

    if multi is None:
        if isinstance(data, tuple):
            multi = True
        else:
            multi = False

    if statfunction is None:
        if _iter:
            statfunction = np.average
        else:
            def statfunc_wrapper(x, *args, **kwargs):
                return np.average(x, axis=-1, *args, **kwargs)
            statfunction = statfunc_wrapper
        
    # Ensure that the data is actually an array. This isn't nice to pandas,
    # but pandas seems much much slower and the indexes become a problem.
    if not multi:
        data = np.array(data)
        tdata = (data,)
    else:
        tdata = tuple( np.array(x) for x in data )

    # Deal with ABC *now*, as it doesn't need samples.
    if method == 'abc':
        n = tdata[0].shape[0]*1.0
        nn = tdata[0].shape[0]

        I = np.identity(nn)
        ep = epsilon / n*1.0
        p0 = np.repeat(1.0/n,nn)

        try:
            t0 = statfunction(*tdata,weights=p0)
        except TypeError as e:
            raise TypeError("statfunction does not accept correct arguments for ABC ({0})".format(e.message))

        di_full = I - p0
        tp = np.fromiter((statfunction(*tdata, weights=p0+ep*di)
                          for di in di_full), dtype=np.float)
        tm = np.fromiter((statfunction(*tdata, weights=p0-ep*di)
                          for di in di_full), dtype=np.float)
        t1 = (tp-tm)/(2*ep)
        t2 = (tp-2*t0+tm)/ep**2

        sighat = np.sqrt(np.sum(t1**2))/n
        a = (np.sum(t1**3))/(6*n**3*sighat**3)
        delta = t1/(n**2*sighat)
        cq = (statfunction(*tdata,weights=p0+ep*delta)-2*t0+statfunction(*tdata,weights=p0-ep*delta))/(2*sighat*ep**2)
        bhat = np.sum(t2)/(2*n**2)
        curv = bhat/sighat-cq
        z0 = nppf(2*ncdf(a)*ncdf(-curv))
        Z = z0+nppf(alphas)
        za = Z/(1-a*Z)**2
        # stan = t0 + sighat * nppf(alphas)
        abc = np.zeros_like(alphas)
        for i in range(0,len(alphas)):
            abc[i] = statfunction(*tdata,weights=p0+za[i]*delta)

        if output == 'lowhigh':
            return abc
        elif output == 'errorbar':
            return abs(abc-statfunction(tdata))[np.newaxis].T
        else:
            raise ValueError("Output option {0} is not supported.".format(output))

    # We don't need to generate actual samples; that would take more memory.
    # Instead, we can generate just the indexes, and then apply the statfun
    # to those indexes.
    if _iter:
        bootindexes = bootstrap_indexes(tdata[0], n_samples)
        stat = np.array([statfunction(*(x[indexes] for x in tdata))
                         for indexes in bootindexes])
    else:
        bootindexes = bootstrap_indexes_array(tdata[0], n_samples)
        stat = statfunction(*(x[bootindexes] for x in tdata))

    stat.sort(axis=0)

    # Percentile Interval Method
    if method == 'pi':
        avals = alphas

    # Bias-Corrected Accelerated Method
    elif method == 'bca':

        # The value of the statistic function applied just to the actual data.
        ostat = statfunction(*tdata)

        # The bias correction value.
        z0 = nppf( ( 1.0*np.sum(stat < ostat, axis=0)  ) / n_samples )

        # Statistics of the jackknife distribution
        jackindexes = jackknife_indexes(tdata[0])
        jstat = [statfunction(*(x[indexes] for x in tdata)) for indexes in jackindexes]
        jmean = np.mean(jstat,axis=0)

        # Temporarily kill numpy warnings:
        oldnperr = np.seterr(invalid='ignore')
        # Acceleration value
        a = np.sum((jmean - jstat)**3, axis=0) / (
            6.0 * np.sum((jmean - jstat)**2, axis=0)**1.5)
        if np.any(np.isnan(a)):
            nanind = np.nonzero(np.isnan(a))
            warnings.warn("BCa acceleration values for indexes {} were undefined. \
Statistic values were likely all equal. Affected CI will \
be inaccurate.".format(nanind), InstabilityWarning, stacklevel=2)
        
        zs = z0 + nppf(alphas).reshape(alphas.shape+(1,)*z0.ndim)

        avals = ncdf(z0 + zs/(1-a*zs))
        np.seterr(**oldnperr)
    else:
        raise ValueError("Method {0} is not supported.".format(method))

    nvals = np.round((n_samples-1)*avals)
    
    oldnperr = np.seterr(invalid='ignore')
    if np.any(np.isnan(nvals)):
        warnings.warn("Some values were NaN; results are probably unstable " +
                      "(all values were probably equal)", InstabilityWarning,
                      stacklevel=2)
    if np.any(nvals == 0) or np.any(nvals == n_samples-1):
        warnings.warn("Some values used extremal samples; " +
                      "results are probably unstable.",
                      InstabilityWarning, stacklevel=2)
    elif np.any(nvals < 10) or np.any(nvals >= n_samples-10):
        warnings.warn("Some values used top 10 low/high samples; " +
                      "results may be unstable.",
                      InstabilityWarning, stacklevel=2)
    np.seterr(**oldnperr)
    
    nvals = np.nan_to_num(nvals).astype('int')

    if output == 'lowhigh':
        if nvals.ndim == 1:
            # All nvals are the same. Simple broadcasting
            return stat[nvals]
        else:
            # Nvals are different for each data point. Not simple broadcasting.
            # Each set of nvals along axis 0 corresponds to the data at the same
            # point in other axes.
            return stat[(nvals, np.indices(nvals.shape)[1:].squeeze())]
    elif output == 'errorbar':
        if nvals.ndim == 1:
            return abs(statfunction(data)-stat[nvals])[np.newaxis].T
        else:
            return abs(statfunction(data)-stat[(nvals, np.indices(nvals.shape)[1:])])[np.newaxis].T
    else:
        raise ValueError("Output option {0} is not supported.".format(output))
    
    


def bootstrap_indexes(data, n_samples=10000):
    """
Given data points data, where axis 0 is considered to delineate points, return
an generator for sets of bootstrap indexes. This can be used as a list
of bootstrap indexes (with list(bootstrap_indexes(data))) as well.
    """
    for _ in xrange(n_samples):
        yield randint(data.shape[0], size=(data.shape[0],))

        
def bootstrap_indexes_array(data, n_samples=10000):
    return randint(data.shape[0], size=(n_samples, data.shape[0]))


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
        size = int(round(size*len(data)))
    elif size > 1:
        pass
    else:
        raise ValueError("size cannot be {0}".format(size))
    base = np.tile(np.arange(len(data)),(n_samples,1))
    for sample in base: np.random.shuffle(sample)
    return base[:,0:size]


def bootstrap_indexes_moving_block(data, n_samples=10000,
                                   block_length=3, wrap=False):
    """Generate moving-block bootstrap samples.

Given data points `data`, where axis 0 is considered to delineate points,
return a generator for sets of bootstrap indexes. This can be used as a
list of bootstrap indexes (with list(bootstrap_indexes_moving_block(data))) as
well.

Parameters
----------

n_samples [default 10000]: the number of subsamples to generate.

block_length [default 3]: the length of block.

wrap [default False]: if false, choose only blocks within the data, making
the last block for data of length L start at L-block_length.  If true, choose
blocks starting anywhere, and if they extend past the end of the data, wrap
around to the beginning of the data again.
"""
    n_obs = data.shape[0]
    n_blocks = int(ceil(n_obs / block_length))
    nexts = np.repeat(np.arange(0, block_length)[None, :], n_blocks, axis=0)

    if wrap:
        last_block = n_obs
    else:
        last_block = n_obs - block_length
    
    for _ in xrange(n_samples):
        blocks = np.random.randint(0, last_block, size=n_blocks)
        if not wrap:
            yield (blocks[:, None]+nexts).ravel()[:n_obs]
        else:
            yield np.mod((blocks[:, None]+nexts).ravel()[:n_obs], n_obs)
