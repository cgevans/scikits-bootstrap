from typing import Any, Sequence

import scikits.bootstrap as boot
import numpy as np
from numpy.testing import assert_raises, assert_allclose
import pytest

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class TestCI:
    def setup(self) -> None:
        self.data = np.array(
            [
                1.34016346,
                1.73759123,
                1.49898834,
                -0.22864333,
                2.031034,
                2.17032495,
                1.59645265,
                -0.76945156,
                0.56605824,
                -0.11927018,
                -0.1465108,
                -0.79890338,
                0.77183278,
                -0.82819136,
                1.32667483,
                1.05986776,
                2.14408873,
                -1.43464512,
                2.28743654,
                0.42864858,
            ]
        )
        self.x = [1, 2, 3, 4, 5, 6]
        self.y = [2, 1, 2, 5, 1, 2]
        self.z = [2, 1, 1, -1, -1, -4, -8]
        self.seed = 1234567890
        if PANDAS_AVAILABLE:
            self.pds = pd.Series(self.data, index=np.arange(50, 70))

    @pytest.mark.skipif(
        not boot.bootstrap.NUMBA_AVAILABLE, reason="Numba not available"
    )
    def test_numba_close(self) -> None:
        dat = np.random.randint(10, size=50)
        no_numba = boot.ci(dat, n_samples=100000)
        with_numba = boot.ci(dat, n_samples=100000, use_numba=True)
        assert_allclose(no_numba, with_numba, rtol=1e-2)

    @pytest.mark.skipif(boot.bootstrap.NUMBA_AVAILABLE, reason="Numba is available")
    def test_numba_unavailable(self) -> None:
        with pytest.raises(ValueError):
            boot.ci(self.data, n_samples=100000, use_numba=True)

    def test_bootstrap_indices(self) -> None:
        indices = np.array(
            [
                x
                for x in boot.bootstrap_indices(
                    np.array([1, 2, 3, 4, 5]), n_samples=3, seed=self.seed
                )
            ]
        )
        assert_allclose(
            indices, np.array([[2, 3, 0, 0, 2], [2, 3, 3, 0, 3], [0, 0, 2, 4, 2]])
        )

    def test_bootstrap_indices_moving_block(self) -> None:
        indices = np.array(
            [
                x
                for x in boot.bootstrap_indices_moving_block(
                    np.array([1, 2, 3, 4, 5]), n_samples=3, seed=self.seed
                )
            ]
        )
        assert_allclose(
            indices, np.array([[0, 1, 2, 1, 2], [0, 1, 2, 0, 1], [1, 2, 3, 1, 2]])
        )

    def test_jackknife_indices(self) -> None:
        indices = np.array([x for x in boot.jackknife_indices(np.array([1, 2, 3]))])
        assert_allclose(indices, np.array([[1, 2], [0, 2], [0, 1]]))

    def test_subsample_indices(self) -> None:
        indices = boot.subsample_indices(self.data, 1000, 0.5)
        # Each sample when sorted must contain len(self.data)/2 unique numbers (eg, be entirely unique)
        for x in indices:
            np.testing.assert_(len(np.unique(x)) == len(self.data) / 2)

    def test_subsample_indices_notsame(self) -> None:
        indices = boot.subsample_indices(np.arange(0, 50), 1000, -1)
        # Test to make sure that subsamples are not all the same.
        # In theory, this test could fail even with correct code, but in
        # practice the probability is too low to care, and the test is useful.
        np.testing.assert_(not np.all(indices[0] == indices[1:]))

    def test_abc_simple(self) -> None:
        results = boot.ci(self.data, method="abc", seed=self.seed)
        assert_allclose(results, np.array([0.20982275, 1.20374686]))

    def test_abc_multialpha_defaultstat(self) -> None:
        results = boot.ci(
            self.data, alpha=(0.1, 0.2, 0.8, 0.9), method="abc", seed=self.seed
        )
        assert_allclose(
            results, np.array([0.39472915, 0.51161304, 0.93789723, 1.04407254])
        )

    # I can't actually figure out how to make this work right now...
    #    def test_abc_epsilon(self) -> None:
    #        results = boot.ci_abc(self.data,lambda x,y: np.sum(y*np.sin(100*x))/
    # np.sum(y),alpha=(0.1,0.2,0.8,0.9))
    #        assert_allclose(results,np.array([-0.11925356, -0.03973595,
    # 0.24915691,  0.32083297]))
    #        results = boot.ci_abc(self.data,lambda x,y: np.sum(y*np.sin(100*x))/
    # np.sum(y),alpha=(0.1,0.2,0.8,0.9),epsilon=20000.5)
    #        assert_allclose(results,np.array([-0.11925356, -0.03973595,
    # 0.24915691,  0.32083297]))

    def test_pi_multialpha(self) -> None:
        results = boot.ci(
            self.data, method="pi", alpha=(0.1, 0.2, 0.8, 0.9), seed=self.seed
        )
        assert_allclose(
            results, np.array([0.401879, 0.517506, 0.945416, 1.052798]), rtol=1e-6
        )

    def test_bca_simple(self) -> None:
        results = boot.ci(self.data, seed=self.seed)
        results2 = boot.ci(self.data, alpha=(0.025, 1 - 0.025), seed=self.seed)
        assert_allclose(results, results2)

    def test_bca_errorbar_output_simple(self) -> None:
        results_default = boot.ci(self.data, seed=self.seed)
        results_errorbar = boot.ci(self.data, output="errorbar", seed=self.seed)
        assert_allclose(
            results_errorbar.T, abs(np.average(self.data) - results_default)[np.newaxis]
        )

    def test_abc_errorbar_output_simple(self) -> None:
        results_default = boot.ci(self.data, method="abc")
        results_errorbar = boot.ci(self.data, output="errorbar", method="abc")
        assert_allclose(
            results_errorbar.T, abs(np.average(self.data) - results_default)[np.newaxis]
        )

    def test_bca_multialpha(self) -> None:
        results = boot.ci(self.data, alpha=(0.1, 0.2, 0.8, 0.9), seed=self.seed)
        assert_allclose(
            results, np.array([0.386674, 0.506714, 0.935628, 1.039683]), rtol=1e-6
        )

    def test_bca_multi_multialpha(self) -> None:
        def statfun(a: Sequence[Any], b: Sequence[Any]) -> Any:
            return np.polyfit(a, b, 1)

        results1 = boot.ci(
            (self.x, self.y),
            statfun,
            alpha=(0.1, 0.2, 0.8, 0.9),
            n_samples=1000,
            seed=self.seed,
        )
        results2 = boot.ci(
            np.vstack((self.x, self.y)).T,
            lambda a: np.polyfit(a[:, 0], a[:, 1], 1),
            alpha=(0.1, 0.2, 0.8, 0.9),
            n_samples=1000,
            seed=self.seed,
        )
        results3 = boot.ci(
            (self.x, self.y),
            statfun,
            alpha=(0.1, 0.2, 0.8, 0.9),
            n_samples=1000,
            seed=self.seed,
            output="errorbar",
        )
        assert_allclose(results1, results2)
        assert_allclose(np.abs(statfun(self.x, self.y) - results1), results3.T)

    def test_bca_multi_indep(self) -> None:
        results1 = boot.ci(
            (self.x, self.z),
            lambda a, b: np.average(a) - np.average(b),
            n_samples=1000,
            multi="independent",
            seed=self.seed,
        )
        assert_allclose(results1, np.array([2.547619, 7.97619]))

    def test_bca_multi_unequal_paired(self) -> None:
        with pytest.raises(ValueError):
            boot.ci(
                (self.x, self.z),
                lambda a, b: np.average(a) - np.average(b),
                n_samples=1000,
                multi="paired",
                seed=self.seed,
            )

    def test_bca_multi_2dout_multialpha(self) -> None:
        results1 = boot.ci(
            (self.x, self.y),
            lambda a, b: np.polyfit(a, b, 1),
            alpha=(0.1, 0.2, 0.8, 0.9),
            n_samples=2000,
            seed=self.seed,
        )
        results2 = boot.ci(
            np.vstack((self.x, self.y)).T,
            lambda a: np.polyfit(a[:, 0], a[:, 1], 1)[0],
            alpha=(0.1, 0.2, 0.8, 0.9),
            n_samples=2000,
            seed=self.seed,
        )
        results3 = boot.ci(
            np.vstack((self.x, self.y)).T,
            lambda a: np.polyfit(a[:, 0], a[:, 1], 1)[1],
            alpha=(0.1, 0.2, 0.8, 0.9),
            n_samples=2000,
            seed=self.seed,
        )
        assert_allclose(results1[:, 0], results2)
        assert_allclose(results1[:, 1], results3)

    def test_multi_fail(self) -> None:
        assert_raises(
            ValueError,
            boot.ci,
            (self.x, self.z),
            lambda a, b: np.average(a) - np.average(b),
            n_samples=1000,
            multi="indepedent",
        )

    def test_non_callable(self) -> None:
        with pytest.raises(TypeError):
            boot.ci(self.data, "average")  # type: ignore

    def test_abc_with_returndist(self) -> None:
        with pytest.raises(ValueError):
            ci, dist = boot.ci(self.data, method="abc", return_dist=True)

    def test_pi_multi_2dout_multialpha(self) -> None:
        results1 = boot.ci(
            (self.x, self.y),
            lambda a, b: np.polyfit(a, b, 1),
            alpha=(0.1, 0.2, 0.8, 0.9),
            n_samples=2000,
            method="pi",
            seed=self.seed,
        )
        results2 = boot.ci(
            np.vstack((self.x, self.y)).T,
            lambda a: np.polyfit(a[:, 0], a[:, 1], 1)[0],
            alpha=(0.1, 0.2, 0.8, 0.9),
            n_samples=2000,
            method="pi",
            seed=self.seed,
        )
        results3 = boot.ci(
            np.vstack((self.x, self.y)).T,
            lambda a: np.polyfit(a[:, 0], a[:, 1], 1)[1],
            alpha=(0.1, 0.2, 0.8, 0.9),
            n_samples=2000,
            method="pi",
            seed=self.seed,
        )
        assert_allclose(results1[:, 0], results2)
        assert_allclose(results1[:, 1], results3)

    def test_bca_n_samples(self) -> None:
        results = boot.ci(
            self.data, alpha=(0.1, 0.2, 0.8, 0.9), n_samples=500, seed=self.seed
        )
        assert_allclose(
            results, np.array([0.37248, 0.507976, 0.92783, 1.039755]), rtol=1e-6
        )

    def test_pi_simple(self) -> None:
        results = boot.ci(self.data, method="pi", seed=self.seed)
        results2 = boot.ci(
            self.data, method="pi", alpha=(0.025, 1 - 0.025), seed=self.seed
        )
        assert_allclose(results, results2)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_abc_pandas_series(self) -> None:
        results = boot.ci(self.pds, method="abc", seed=self.seed)
        results2 = boot.ci(self.data, method="abc", seed=self.seed)
        assert_allclose(results, results2)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_bca_pandas_series(self) -> None:
        results = boot.ci(self.pds, seed=self.seed)
        results2 = boot.ci(self.data, seed=self.seed)
        assert_allclose(results, results2)

    @pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not available")
    def test_pi_pandas_series(self) -> None:
        results = boot.ci(self.pds, method="pi", seed=self.seed)
        results2 = boot.ci(self.data, method="pi", seed=self.seed)
        assert_allclose(results, results2)

    def test_invalid_multi(self) -> None:
        with pytest.raises(
            ValueError, match=r"Value `wrong` for multi is not recognized."
        ):
            boot.ci(self.data, multi="wrong")  # type: ignore

    def test_pval(self) -> None:
        result = boot.pval(
            self.data,
            np.average,
            lambda s: 0.8 <= s <= 1.2,
            n_samples=500,
            seed=self.seed,
        )
        assert_allclose(result, 0.368)

    def test_pval_implicit_and_explicit_multi(self) -> None:
        result = boot.pval(
            (self.x, self.y),
            lambda x, y: np.array([np.average(x), np.average(y)]),
            lambda s: s <= 3,
            n_samples=500,
            seed=self.seed,
        )
        result2 = boot.pval(
            (self.x, self.y),
            lambda x, y: np.array([np.average(x), np.average(y)]),
            lambda s: s <= 3,
            n_samples=500,
            seed=self.seed,
            multi=True,
        )
        assert_allclose(result, [0.262, 0.936])
        assert_allclose(result, result2)
