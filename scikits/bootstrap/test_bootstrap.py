from __future__ import division

import scikits.bootstrap as boot
import numpy as np
from numpy.testing import dec, assert_raises


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
        self.z = [2,1,1,-1,-1,-4,-8]
        self.seed = 1234567890
        if not no_pandas:
            self.pds = pandas.Series(self.data,index=np.arange(50,70))

    def test_bootstrap_indices(self):
        indices = np.array([x for x in boot.bootstrap_indices(
            np.array([1, 2, 3, 4, 5]), n_samples=3, seed=self.seed)])
        np.testing.assert_array_equal(indices, np.array([[2, 3, 0, 0, 2],
                                                         [2, 3, 3, 0, 3],
                                                         [0, 0, 2, 4, 2]])
                                      )

    def test_bootstrap_indices_moving_block(self):
        indices = np.array([x for x in boot.bootstrap_indices_moving_block(
            np.array([1, 2, 3, 4, 5]), n_samples=3, seed=self.seed)])
        np.testing.assert_array_equal(indices, np.array([[0, 1, 2, 1, 2],
             [0, 1, 2, 0, 1],
             [1, 2, 3, 1, 2]]))

    def test_jackknife_indices(self):
        indices = np.array([x for x in boot.jackknife_indices(np.array([1,2,3]))])
        np.testing.assert_array_equal(indices, np.array([[1, 2],[0, 2],[0, 1]]))

    def test_subsample_indices(self):
        indices = boot.subsample_indices(self.data, 1000, 0.5)
        # Each sample when sorted must contain len(self.data)/2 unique numbers (eg, be entirely unique)
        for x in indices:
            np.testing.assert_(len(np.unique(x)) == len(self.data)/2)

    def test_subsample_indices_notsame(self):
        indices = boot.subsample_indices(np.arange(0,50), 1000, -1)
        # Test to make sure that subsamples are not all the same.
        # In theory, this test could fail even with correct code, but in
        # practice the probability is too low to care, and the test is useful.
        np.testing.assert_(not np.all(indices[0]==indices[1:]))

    def test_abc_simple(self):
        results = boot.ci(self.data,
                          lambda x, weights: np.average(x, weights=weights),
                          method='abc', seed=self.seed)
        np.testing.assert_array_almost_equal(
            results, np.array([0.20982275, 1.20374686]))
    
    def test_abc_multialpha_defaultstat(self):
        results = boot.ci(self.data, alpha=(
            0.1, 0.2, 0.8, 0.9), method='abc', seed=self.seed)
        np.testing.assert_array_almost_equal(results,np.array([ 0.39472915,  0.51161304,  0.93789723,  1.04407254]))

# I can't actually figure out how to make this work right now...
#    def test_abc_epsilon(self):
#        results = boot.ci_abc(self.data,lambda x,y: np.sum(y*np.sin(100*x))/np.sum(y),alpha=(0.1,0.2,0.8,0.9))
#        np.testing.assert_array_almost_equal(results,np.array([-0.11925356, -0.03973595,  0.24915691,  0.32083297]))
#        results = boot.ci_abc(self.data,lambda x,y: np.sum(y*np.sin(100*x))/np.sum(y),alpha=(0.1,0.2,0.8,0.9),epsilon=20000.5)
#        np.testing.assert_array_almost_equal(results,np.array([-0.11925356, -0.03973595,  0.24915691,  0.32083297]))

    def test_pi_multialpha(self):
        results = boot.ci(self.data,
                          method='pi',
                          alpha=(0.1, 0.2, 0.8, 0.9),
                          seed=self.seed)
        np.testing.assert_array_almost_equal(
            results, np.array([0.401879, 0.517506, 0.945416, 1.052798]))

    def test_bca_simple(self):
        results = boot.ci(self.data, seed=self.seed)
        results2 = boot.ci(self.data, alpha=(0.025, 1-0.025), seed=self.seed)
        np.testing.assert_array_almost_equal(results, results2)

    def test_bca_errorbar_output_simple(self):
        results_default = boot.ci(self.data, seed=self.seed)
        results_errorbar = boot.ci(
            self.data, output='errorbar', seed=self.seed)
        np.testing.assert_array_almost_equal(
            results_errorbar.T,
            abs(np.average(self.data) - results_default)[np.newaxis])

    def test_bca_multialpha(self):
        results = boot.ci(self.data, alpha=(
            0.1, 0.2, 0.8, 0.9), seed=self.seed)
        np.testing.assert_array_almost_equal(results, np.array(
            [0.386674, 0.506714, 0.935628, 1.039683]))
        
    def test_bca_multi_multialpha(self):
        results1 = boot.ci((self.x, self.y), lambda a, b: np.polyfit(
            a, b, 1), alpha=(0.1, 0.2, 0.8, 0.9), n_samples=1000, seed=self.seed)
        results2 = boot.ci(np.vstack((self.x, self.y)).T, lambda a: np.polyfit(
            a[:, 0], a[:, 1], 1), alpha=(0.1, 0.2, 0.8, 0.9), n_samples=1000, seed=self.seed)
        np.testing.assert_array_almost_equal(results1,results2)

    def test_bca_multi_indep(self):
        results1 = boot.ci((self.x, self.z), lambda a,b: np.average(a) - np.average(b), n_samples=1000, multi='independent')

    def test_bca_multi_2dout_multialpha(self):
        results1 = boot.ci((self.x, self.y), lambda a, b: np.polyfit(
            a, b, 1), alpha=(0.1, 0.2, 0.8, 0.9), n_samples=2000, seed=self.seed)
        results2 = boot.ci(np.vstack((self.x, self.y)).T, lambda a: np.polyfit(
            a[:, 0], a[:, 1], 1)[0], alpha=(0.1, 0.2, 0.8, 0.9), n_samples=2000, seed=self.seed)
        results3 = boot.ci(np.vstack((self.x, self.y)).T, lambda a: np.polyfit(
            a[:, 0], a[:, 1], 1)[1], alpha=(0.1, 0.2, 0.8, 0.9), n_samples=2000, seed=self.seed)
        np.testing.assert_array_almost_equal(results1[:,0],results2)
        np.testing.assert_array_almost_equal(results1[:,1],results3)

    def test_multi_fail(self):
        assert_raises(ValueError,
        boot.ci,
        (self.x,self.z), lambda a,b: np.average(a) - np.average(b), n_samples=1000, multi='indepedent')

    def test_pi_multi_2dout_multialpha(self):
        results1 = boot.ci((self.x, self.y), lambda a, b: np.polyfit(a, b, 1), alpha=(
            0.1, 0.2, 0.8, 0.9), n_samples=2000, method='pi', seed=self.seed)
        results2 = boot.ci(np.vstack((self.x, self.y)).T, lambda a: np.polyfit(a[:, 0], a[:, 1], 1)[
                           0], alpha=(0.1, 0.2, 0.8, 0.9), n_samples=2000, method='pi', seed=self.seed)
        results3 = boot.ci(np.vstack((self.x, self.y)).T, lambda a: np.polyfit(a[:, 0], a[:, 1], 1)[
                           1], alpha=(0.1, 0.2, 0.8, 0.9), n_samples=2000, method='pi', seed=self.seed)
        np.testing.assert_array_almost_equal(results1[:,0],results2)
        np.testing.assert_array_almost_equal(results1[:,1],results3)
    
    def test_bca_n_samples(self):
        seed = self.seed
        results = boot.ci(self.data,
                          alpha=(0.1, 0.2, 0.8, 0.9),
                          n_samples=500,
                          seed=self.seed)
        np.testing.assert_array_almost_equal(
            results, np.array([0.37248, 0.507976, 0.92783, 1.039755]))

    def test_pi_simple(self):
        seed = self.seed
        results = boot.ci(self.data, method='pi', seed=self.seed)
        results2 = boot.ci(self.data, method='pi',
                           alpha=(0.025, 1-0.025), seed=self.seed)
        np.testing.assert_array_almost_equal(results, results2)

    @dec.skipif(no_pandas)
    def test_abc_pandas_series(self):
        results = boot.ci(self.pds, method='abc', seed=self.seed)
        results2 = boot.ci(self.data, method='abc', seed=self.seed)
        np.testing.assert_array_almost_equal(results, results2)

    @dec.skipif(no_pandas)
    def test_bca_pandas_series(self):
        results = boot.ci(self.pds, seed=self.seed)
        results2 = boot.ci(self.data, seed=self.seed)
        np.testing.assert_array_almost_equal(results, results2)

    @dec.skipif(no_pandas)
    def test_pi_pandas_series(self):
        results = boot.ci(self.pds, method='pi', seed=self.seed)
        results2 = boot.ci(self.data, method='pi', seed=self.seed)
        np.testing.assert_array_almost_equal(results, results2)

class test_pval():
    def setup(self):
        self.seed = 123467890

    def test_pval(self):
        rng = np.random.default_rng(seed=self.seed)
        mu = 1
        s2 = 2
        N = 10000
        NS = 10000

        data = rng.normal(mu, s2, N)

        # print "Dist Normal(%.1f, %.1f)" % (mu,s2)
        # print "Analytic CI: ", [np.average(data) - 1.96 * np.sqrt(np.var(data)) / np.sqrt(N), np.average(data) + 1.96 * np.sqrt(np.var(data)) / np.sqrt(N) ]
        # print "Bootstrap CI:", list(bs.ci(data, np.average, n_samples = NS))

        # print "P(np.average is in 95% CI):", bs.pval(data, np.average, lambda s: 0.98544817 <= s <= 1.06404872, n_samples=NS)

        result = boot.pval(data, np.average, lambda s: 0.98544817 <=
                           s <= 1.06404872, n_samples=NS, seed=rng)
        np.testing.assert_almost_equal(result, 0.95079, 3)

if __name__ == "__main__":
    np.testing.run_module_suite()
