from __future__ import division

import scikits.bootstrap as boot
import numpy as np
from numpy.testing.decorators import skipif


try:
    import pandas
except ImportError:
    no_pandas = True
else:
    no_pandas = False

class test_ci():
    def setup(self):
        self.data = np.array([ 1.34016346,  1.73759123,  1.49898834, -0.22864333,  2.031034  ,
                            2.17032495,  1.59645265, -0.76945156,  0.56605824, -0.11927018,
                           -0.1465108 , -0.79890338,  0.77183278, -0.82819136,  1.32667483,
                            1.05986776,  2.14408873, -1.43464512,  2.28743654,  0.42864858])
        self.x = [1,2,3,4,5,6]
        self.y = [2,1,2,5,1,2]
        if not no_pandas:
            self.pds = pandas.Series(self.data,index=np.arange(50,70))

    def test_bootstrap_indexes(self):
        np.random.seed(1234567890)
        indexes = np.array([x for x in boot.bootstrap_indexes(np.array([1,2,3,4,5]), n_samples=3)])
        np.testing.assert_array_equal(indexes, np.array([[2, 4, 3, 1, 3],[1, 4, 1, 4, 4],[0, 2, 1, 4, 4]]))

    def test_bootstrap_indexes_moving_block(self):
        np.random.seed(1234567897)
        indexes = np.array([x for x in boot.bootstrap_indexes_moving_block(np.array([1,2,3,4,5]), n_samples=3)])
        np.testing.assert_array_equal(indexes, np.array([[1, 2, 3, 1, 2], [0, 1, 2, 0, 1], [0, 1, 2, 0, 1]]))

    def test_jackknife_indexes(self):
        np.random.seed(1234567890)
        indexes = np.array([x for x in boot.jackknife_indexes(np.array([1,2,3]))])
        np.testing.assert_array_equal(indexes, np.array([[1, 2],[0, 2],[0, 1]]))

    def test_subsample_indexes(self):
        indexes = boot.subsample_indexes(self.data, 1000, 0.5)
        # Each sample when sorted must contain len(self.data)/2 unique numbers (eg, be entirely unique)
        for x in indexes:
            np.testing.assert_(len(np.unique(x)) == len(self.data)/2)

    def test_subsample_indexes_notsame(self):
        np.random.seed(1234567890)
        indexes = boot.subsample_indexes(np.arange(0,50), 1000, -1)
        # Test to make sure that subsamples are not all the same.
        # In theory, this test could fail even with correct code, but in
        # practice the probability is too low to care, and the test is useful.
        np.testing.assert_(not np.all(indexes[0]==indexes[1:]))

    def test_abc_simple(self):
        results = boot.ci(self.data,
                          lambda x, weights: np.average(x, weights=weights),
                          method='abc')
        np.testing.assert_array_almost_equal(
            results, np.array([0.20982275, 1.20374686]))

    def test_abc_multialpha_unified_noiter(self):
        results = boot.ci(self.data,
                          lambda x, weights:
                          np.average(x, weights=weights, axis=-1),
                          alpha=(0.1, 0.2, 0.8, 0.9), method='abc', _iter=False)
        np.testing.assert_array_almost_equal(
            results,
            np.array([0.39472915, 0.51161304, 0.93789723, 1.04407254]))
    
    def test_abc_multialpha_defaultstat(self):
        results = boot.ci(self.data, alpha=(0.1,0.2,0.8,0.9), method='abc')
        np.testing.assert_array_almost_equal(results,np.array([ 0.39472915,  0.51161304,  0.93789723,  1.04407254]))

# I can't actually figure out how to make this work right now...
#    def test_abc_epsilon(self):
#        results = boot.ci_abc(self.data,lambda x,y: np.sum(y*np.sin(100*x))/np.sum(y),alpha=(0.1,0.2,0.8,0.9))
#        np.testing.assert_array_almost_equal(results,np.array([-0.11925356, -0.03973595,  0.24915691,  0.32083297]))
#        results = boot.ci_abc(self.data,lambda x,y: np.sum(y*np.sin(100*x))/np.sum(y),alpha=(0.1,0.2,0.8,0.9),epsilon=20000.5)
#        np.testing.assert_array_almost_equal(results,np.array([-0.11925356, -0.03973595,  0.24915691,  0.32083297]))

    def test_pi_multialpha(self):
        np.random.seed(1234567890)
        results = boot.ci(self.data, method='pi', alpha=(0.1,0.2,0.8,0.9))
        np.testing.assert_array_almost_equal(results,np.array([ 0.40351601,  0.51723236,  0.94547054,  1.05749207]))

    def test_bca_simple(self):
        np.random.seed(1234567890)
        results = boot.ci(self.data)
        np.testing.assert_array_almost_equal(
            results, np.array([0.20907826, 1.19877862]))

    def test_bca_errorbar_output_simple(self):
        np.random.seed(1234567890)
        results_default = boot.ci(self.data)
        np.random.seed(1234567890)
        results_errorbar = boot.ci(self.data, output='errorbar')
        np.testing.assert_array_almost_equal(
            results_errorbar.T,
            abs(np.average(self.data) - results_default)[np.newaxis])

    def test_bca_multialpha(self):
        np.random.seed(1234567890)
        results = boot.ci(self.data, alpha=(0.1, 0.2, 0.8, 0.9))
        np.testing.assert_array_almost_equal(results, np.array(
            [0.39210727, 0.50775386, 0.93673299, 1.0476729]))

    def test_bca_multialpha_noiter(self):
        np.random.seed(1234567890)
        results = boot.ci(self.data, alpha=(0.1, 0.2, 0.8, 0.9), _iter=False)
        np.testing.assert_array_almost_equal(
            results, np.array([0.39210727, 0.50775386, 0.93673299, 1.0476729]))
        
    def test_bca_multi_multialpha(self):
        np.random.seed(1234567890)
        results1 = boot.ci((self.x,self.y), lambda a,b: np.polyfit(a,b,1), alpha=(0.1,0.2,0.8,0.9),n_samples=1000)
        np.random.seed(1234567890)
        results2 = boot.ci(np.vstack((self.x,self.y)).T, lambda a: np.polyfit(a[:,0],a[:,1],1), alpha=(0.1,0.2,0.8,0.9),n_samples=1000)
        np.testing.assert_array_almost_equal(results1,results2)

    def test_bca_multi_2dout_multialpha(self):
        np.random.seed(1234567890)
        results1 = boot.ci((self.x,self.y), lambda a,b: np.polyfit(a,b,1), alpha=(0.1,0.2,0.8,0.9),n_samples=2000)
        np.random.seed(1234567890)
        results2 = boot.ci(np.vstack((self.x,self.y)).T, lambda a: np.polyfit(a[:,0],a[:,1],1)[0], alpha=(0.1,0.2,0.8,0.9),n_samples=2000)
        np.random.seed(1234567890)
        results3 = boot.ci(np.vstack((self.x,self.y)).T, lambda a: np.polyfit(a[:,0],a[:,1],1)[1], alpha=(0.1,0.2,0.8,0.9),n_samples=2000)
        np.testing.assert_array_almost_equal(results1[:,0],results2)
        np.testing.assert_array_almost_equal(results1[:,1],results3)

    def test_pi_multi_2dout_multialpha(self):
        np.random.seed(1234567890)
        results1 = boot.ci((self.x,self.y), lambda a,b: np.polyfit(a,b,1), alpha=(0.1,0.2,0.8,0.9),n_samples=2000,method='pi')
        np.random.seed(1234567890)
        results2 = boot.ci(np.vstack((self.x,self.y)).T, lambda a: np.polyfit(a[:,0],a[:,1],1)[0], alpha=(0.1,0.2,0.8,0.9),n_samples=2000,method='pi')
        np.random.seed(1234567890)
        results3 = boot.ci(np.vstack((self.x,self.y)).T, lambda a: np.polyfit(a[:,0],a[:,1],1)[1], alpha=(0.1,0.2,0.8,0.9),n_samples=2000,method='pi')
        np.testing.assert_array_almost_equal(results1[:,0],results2)
        np.testing.assert_array_almost_equal(results1[:,1],results3)
    
    def test_bca_n_samples(self):
        np.random.seed(1234567890)
        results = boot.ci(self.data, alpha=(0.1,0.2,0.8,0.9),n_samples=500)
        np.testing.assert_array_almost_equal(results,np.array([ 0.40027628,  0.5063184 ,  0.94082515,  1.05653929]))

    def test_pi_simple(self):
        np.random.seed(1234567890)
        results = boot.ci(self.data, method='pi')
        np.testing.assert_array_almost_equal(results,np.array([ 0.2288689 ,  1.21259752]))

    @skipif(no_pandas)
    def test_abc_pandas_series(self):
        results = boot.ci(self.pds, method='abc')
        np.testing.assert_array_almost_equal(results,np.array([ 0.20982275,  1.20374686]))

    @skipif(no_pandas)
    def test_bca_pandas_series(self):
        np.random.seed(1234567890)
        results = boot.ci(self.pds)
        np.testing.assert_array_almost_equal(results,np.array([ 0.20907826,  1.19877862]))

    @skipif(no_pandas)
    def test_pi_pandas_series(self):
        np.random.seed(1234567890)
        results = boot.ci(self.pds, method='pi')
        np.testing.assert_array_almost_equal(results,np.array([ 0.2288689 ,  1.21259752]))

if __name__ == "__main__":
    np.testing.run_module_suite()
