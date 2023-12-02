# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for BaseDistribution API points."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd
import pytest

from sktime.datatypes import check_is_mtype
from sktime.tests.test_all_estimators import BaseFixtureGenerator, QuickTester


class DistributionFixtureGenerator(BaseFixtureGenerator):
    """Fixture generator for probability distributions.

    Fixtures parameterized
    ----------------------
    estimator_class: estimator inheriting from BaseObject
        ranges over estimator classes not excluded by EXCLUDE_ESTIMATORS, EXCLUDED_TESTS
    estimator_instance: instance of estimator inheriting from BaseObject
        ranges over estimator classes not excluded by EXCLUDE_ESTIMATORS, EXCLUDED_TESTS
        instances are generated by create_test_instance class method
    """

    estimator_type_filter = "distribution"


def _has_capability(distr, method):
    """Check whether distr has capability of method.

    Parameters
    ----------
    distr : BaseDistribution object
    method : str
        method name to check

    Returns
    -------
    whether distr has capability method, according to tags
    capabilities:approx and capabilities:exact
    """
    approx_methods = distr.get_tag("capabilities:approx")
    exact_methods = distr.get_tag("capabilities:exact")
    return method in approx_methods or method in exact_methods


METHODS_SCALAR = ["mean", "var", "energy"]
METHODS_SCALAR_POS = ["var", "energy"]  # result always non-negative?
METHODS_X = ["energy", "pdf", "log_pdf", "cdf"]
METHODS_X_POS = ["energy", "pdf", "cdf"]  # result always non-negative?
METHODS_P = ["ppf"]
METHODS_ROWWISE = ["energy"]  # results in one column


class TestAllDistributions(DistributionFixtureGenerator, QuickTester):
    """Module level tests for all sktime parameter fitters."""

    def test_sample(self, estimator_instance):
        """Test sample expected return."""
        d = estimator_instance

        res = d.sample()

        assert d.shape == res.shape
        assert (res.index == d.index).all()
        assert (res.columns == d.columns).all()

        res_panel = d.sample(3)
        dummy_panel = pd.concat([res, res, res], keys=range(3))
        assert dummy_panel.shape == res_panel.shape
        assert (res_panel.index == dummy_panel.index).all()
        assert (res_panel.columns == dummy_panel.columns).all()

    @pytest.mark.parametrize("method", METHODS_SCALAR, ids=METHODS_SCALAR)
    def test_methods_scalar(self, estimator_instance, method):
        """Test expected return of scalar methods."""
        if not _has_capability(estimator_instance, method):
            return None

        d = estimator_instance
        res = getattr(estimator_instance, method)()

        _check_output_format(res, d, method)

    @pytest.mark.parametrize("method", METHODS_X, ids=METHODS_X)
    def test_methods_x(self, estimator_instance, method):
        """Test expected return of methods that take sample-like argument."""
        if not _has_capability(estimator_instance, method):
            return None

        d = estimator_instance
        x = d.sample()
        res = getattr(estimator_instance, method)(x)

        _check_output_format(res, d, method)

    @pytest.mark.parametrize("method", METHODS_P, ids=METHODS_P)
    def test_methods_p(self, estimator_instance, method):
        """Test expected return of methods that take percentage-like argument."""
        if not _has_capability(estimator_instance, method):
            return None

        d = estimator_instance
        np_unif = np.random.uniform(size=d.shape)
        p = pd.DataFrame(np_unif, index=d.index, columns=d.columns)
        res = getattr(estimator_instance, method)(p)

        _check_output_format(res, d, method)

    @pytest.mark.parametrize("q", [0.7, [0.1, 0.3, 0.9]])
    def test_quantile(self, estimator_instance, q):
        """Test expected return of quantile method."""
        if not _has_capability(estimator_instance, "ppf"):
            return None

        d = estimator_instance

        def _check_quantile_output(obj, q):
            assert check_is_mtype(
                obj, "pred_quantiles", "Proba", msg_return_dict="list"
            )
            assert (obj.index == d.index).all()

            if not isinstance(q, list):
                q = [q]
            expected_columns = pd.MultiIndex.from_product([d.columns, q])
            assert (obj.columns == expected_columns).all()

        res = d.quantile(q)
        _check_quantile_output(res, q)


def _check_output_format(res, dist, method):
    """Check output format expectations for BaseDistribution tests."""
    if method in METHODS_ROWWISE:
        exp_shape = (dist.shape[0], 1)
    else:
        exp_shape = dist.shape
    assert res.shape == exp_shape
    assert (res.index == dist.index).all()
    if method not in METHODS_ROWWISE:
        assert (res.columns == dist.columns).all()

    if method in METHODS_SCALAR_POS or method in METHODS_X_POS:
        assert (res >= 0).all().all()
