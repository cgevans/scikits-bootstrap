import scikits.bootstrap as boot
import numpy as np
from numpy.testing.decorators import skipif

try:
    import pandas
except ImportError:
    no_pandas = True
else:
    no_pandas = False

class test_ci:
    def setup(self):
        self.data = np.array([ 1.34016346,  1.73759123,  1.49898834, -0.22864333,  2.031034  ,
                            2.17032495,  1.59645265, -0.76945156,  0.56605824, -0.11927018,
                           -0.1465108 , -0.79890338,  0.77183278, -0.82819136,  1.32667483,
                            1.05986776,  2.14408873, -1.43464512,  2.28743654,  0.42864858])

    def test_bootstrap_indexes(self):
        np.random.seed(1234567890)
        indexes = boot.bootstrap_indexes(np.array([1,2,3,4,5]), n_samples=3)
        np.testing.assert_array_equal(indexes, np.array([[2, 4, 3, 1, 3],[1, 4, 1, 4, 4],[0, 2, 1, 4, 4]]))

    def test_jackknife_indexes(self):
        np.random.seed(1234567890)
        indexes = boot.jackknife_indexes(np.array([1,2,3]))
        np.testing.assert_array_equal(indexes, np.array([[1, 2],[0, 2],[0, 1]]))

    def test_abc_simple(self):
        results = boot.ci_abc(self.data,lambda x,y: np.average(x,weights=y))
        np.testing.assert_array_almost_equal(results,np.array([ 0.20982275,  1.20374686]))

    def test_abc_multialpha(self):
        results = boot.ci_abc(self.data,lambda x,y: np.average(x,weights=y),alpha=(0.1,0.2,0.8,0.9))
        np.testing.assert_array_almost_equal(results,np.array([ 0.39472915,  0.51161304,  0.93789723,  1.04407254]))

# I can't actually figure out how to make this work right now...
#    def test_abc_epsilon(self):
#        results = boot.ci_abc(self.data,lambda x,y: np.sum(y*np.sin(100*x))/np.sum(y),alpha=(0.1,0.2,0.8,0.9))
#        np.testing.assert_array_almost_equal(results,np.array([-0.11925356, -0.03973595,  0.24915691,  0.32083297]))
#        results = boot.ci_abc(self.data,lambda x,y: np.sum(y*np.sin(100*x))/np.sum(y),alpha=(0.1,0.2,0.8,0.9),epsilon=20000.5)
#        np.testing.assert_array_almost_equal(results,np.array([-0.11925356, -0.03973595,  0.24915691,  0.32083297]))

    def test_pi_multialpha(self):
        np.random.seed(1234567890)
        results = boot.ci(self.data,np.average,method='pi',alpha=(0.1,0.2,0.8,0.9))
        np.testing.assert_array_almost_equal(results,np.array([ 0.40351601,  0.51723236,  0.94547519,  1.05757251]))

    def test_bca_simple(self):
        np.random.seed(1234567890)
        results = boot.ci(self.data,np.average)
        np.testing.assert_array_almost_equal(results,np.array([ 0.23335804,  1.21866548]))

    def test_bca_multialpha(self):
        np.random.seed(1234567890)
        results = boot.ci(self.data,np.average,alpha=(0.1,0.2,0.8,0.9))
        np.testing.assert_array_almost_equal(results,np.array([ 0.40082352,  0.51239987,  0.94064136,  1.05528586]))

    def test_bca_n_samples(self):
        np.random.seed(1234567890)
        results = boot.ci(self.data,np.average,alpha=(0.1,0.2,0.8,0.9),n_samples=500)
        np.testing.assert_array_almost_equal(results,np.array([ 0.40351601,  0.51825288,  0.94379501,  1.06723333]))

    def test_pi_simple(self):
        np.random.seed(1234567890)
        results = boot.ci(self.data,np.average,method='pi')
        np.testing.assert_array_almost_equal(results,np.array([ 0.2288689 ,  1.21292773]))

    @skipif(no_pandas)
    def test_abc_pandas_series(self):
        results = boot.ci_abc(pandas.Series(self.data),lambda x,y: np.average(x,weights=y))
        np.testing.assert_array_almost_equal(results,np.array([ 0.20982275,  1.20374686]))

    @skipif(no_pandas)
    def test_bca_pandas_series(self):
        np.random.seed(1234567890)
        results = boot.ci(pandas.Series(self.data),np.average)
        np.testing.assert_array_almost_equal(results,np.array([ 0.23335804,  1.21866548]))

    @skipif(no_pandas)
    def test_pi_pandas_series(self):
        np.random.seed(1234567890)
        results = boot.ci(pandas.Series(self.data),np.average,method='pi')
        np.testing.assert_array_almost_equal(results,np.array([ 0.2288689 ,  1.21292773]))

if __name__ == "__main__":
    np.testing.run_module_suite()
