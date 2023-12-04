#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements Prophet forecaster by wrapping fbprophet."""

__author__ = ["mloning", "aiwalter", "fkiraly", "tpvasconcelos"]
__all__ = ["Prophet"]


import os

import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA


class Prophet(BaseForecaster):
    """Prophet forecaster by wrapping Facebook's prophet algorithm [1]_.

    Direct interface to Facebook prophet, using the sktime interface.
    All hyper-parameters are exposed via the constructor.

    Data can be passed in one of the sktime compatible formats,
    naming a column `ds` such as in the prophet package is not necessary.

    Unlike vanilla `prophet`, also supports integer/range and period index:
    * integer/range index is interpreted as days since Jan 1, 2000
    * `PeriodIndex` is converted using the `pandas` method `to_timestamp`

    Parameters
    ----------
    freq: str, default=None
        A DatetimeIndex frequency. For possible values see
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html
    add_seasonality: dict or None, default=None
        Dict with args for Prophet.add_seasonality().
        Dict can have the following keys/values:
            * name: string name of the seasonality component.
            * period: float number of days in one period.
            * fourier_order: int number of Fourier components to use.
            * prior_scale: optional float prior scale for this component.
            * mode: optional 'additive' or 'multiplicative'
            * condition_name: string name of the seasonality condition.
    add_country_holidays: dict or None, default=None
        Dict with args for Prophet.add_country_holidays().
        Dict can have the following keys/values:
            country_name: Name of the country, like 'UnitedStates' or 'US'
    growth: str, default="linear"
        String 'linear' or 'logistic' to specify a linear or logistic
        trend. If 'logistic' specified float for 'growth_cap' must be provided.
    growth_floor: float, default=0
        Growth saturation minimum value.
        Used only if `growth="logistic"`, has no effect otherwise
        (if `growth` is not `"logistic"`).
    growth_cap: float, default=None
        Growth saturation maximum aka carrying capacity.
        Mandatory (float) iff `growth="logistic"`, has no effect and is optional,
        otherwise (if `growth` is not `"logistic"`).
    changepoints: list or None, default=None
        List of dates at which to include potential changepoints. If
        not specified, potential changepoints are selected automatically.
    n_changepoints: int, default=25
        Number of potential changepoints to include. Not used
        if input `changepoints` is supplied. If `changepoints` is not supplied,
        then n_changepoints potential changepoints are selected uniformly from
        the first `changepoint_range` proportion of the history.
    changepoint_range: float, default=0.8
        Proportion of history in which trend changepoints will
        be estimated. Defaults to 0.8 for the first 80%. Not used if
        `changepoints` is specified.
    yearly_seasonality: str or bool or int, default="auto"
        Fit yearly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    weekly_seasonality: str or bool or int, default="auto"
        Fit weekly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    daily_seasonality: str or bool or int, default="auto"
        Fit daily seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    holidays: pd.DataFrame or None, default=None
        pd.DataFrame with columns holiday (string) and ds (date type)
        and optionally columns lower_window and upper_window which specify a
        range of days around the date to be included as holidays.
        lower_window=-2 will include 2 days prior to the date as holidays. Also
        optionally can have a column prior_scale specifying the prior scale for
        that holiday.
    seasonality_mode: str, default='additive'
        Take one of 'additive' or 'multiplicative'.
    seasonality_prior_scale: float, default=10.0
        Parameter modulating the strength of the seasonality model.
        Larger values allow the model to fit larger seasonal
        fluctuations, smaller values dampen the seasonality. Can be specified
        for individual seasonalities using add_seasonality.
    holidays_prior_scale: float, default=10.0
        Parameter modulating the strength of the holiday
        components model, unless overridden in the holidays input.
    changepoint_prior_scale: float, default=0.05
        Parameter modulating the flexibility of the
        automatic changepoint selection. Large values will allow many
        changepoints, small values will allow few changepoints.
    mcmc_samples: int, default=0
        If greater than 0, will do full Bayesian inference
        with the specified number of MCMC samples. If 0, will do MAP
        estimation.
    alpha: float, default=0.05
        Width of the uncertainty intervals provided
        for the forecast. If mcmc_samples=0, this will be only the uncertainty
        in the trend using the MAP estimate of the extrapolated generative
        model. If mcmc.samples>0, this will be integrated over all model
        parameters, which will include uncertainty in seasonality.
    uncertainty_samples: int, default=1000
        Number of simulated draws used to estimate uncertainty intervals.
        Settings this value to 0 or False will disable
        uncertainty estimation and speed up the calculation.
    stan_backend: str or None, default=None
        str as defined in StanBackendEnum. If None, will try to
        iterate over all available backends and find the working one.
    fit_kwargs: dict or None, default=None
        Dict with args for Prophet.fit().
        These are additional arguments passed to the optimizing or sampling
        functions in Stan.

    References
    ----------
    .. [1] https://facebook.github.io/prophet

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.fbprophet import Prophet
    >>> # Prophet requires to have data with a pandas.DatetimeIndex
    >>> y = load_airline().to_timestamp(freq='M')
    >>> forecaster = Prophet(  # doctest: +SKIP
    ...     seasonality_mode='multiplicative',
    ...     n_changepoints=int(len(y) / 12),
    ...     add_country_holidays={'country_name': 'Germany'},
    ...     yearly_seasonality=True)
    >>> forecaster.fit(y)  # doctest: +SKIP
    Prophet(...)
    >>> y_pred = forecaster.predict(fh=[1,2,3])  # doctest: +SKIP
    """

    _tags = {
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "y_inner_mtype": "pd.DataFrame",
        "python_dependencies": "prophet",
    }

    def __init__(
        self,
        # Args due to wrapping
        freq=None,
        add_seasonality=None,
        add_country_holidays=None,
        # Args of fbprophet
        growth="linear",
        growth_floor=0.0,
        growth_cap=None,
        changepoints=None,
        n_changepoints=25,
        changepoint_range=0.8,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        holidays=None,
        seasonality_mode="additive",
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        changepoint_prior_scale=0.05,
        mcmc_samples=0,
        alpha=DEFAULT_ALPHA,
        uncertainty_samples=1000,
        stan_backend=None,
        verbose=0,
        fit_kwargs=None,
    ):
        self.freq = freq
        self.add_seasonality = add_seasonality
        self.add_country_holidays = add_country_holidays

        self.growth = growth
        self.growth_floor = growth_floor
        self.growth_cap = growth_cap
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = seasonality_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.mcmc_samples = mcmc_samples
        self.alpha = alpha
        self.uncertainty_samples = uncertainty_samples
        self.stan_backend = stan_backend
        self.verbose = verbose
        self.fit_kwargs = fit_kwargs

        super().__init__()

        # import inside method to avoid hard dependency
        from prophet.forecaster import Prophet as _Prophet

        self._ModelClass = _Prophet

    def _instantiate_model(self):
        self._forecaster = self._ModelClass(
            growth=self.growth,
            changepoints=self.changepoints,
            n_changepoints=self.n_changepoints,
            changepoint_range=self.changepoint_range,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            holidays=self.holidays,
            seasonality_mode=self.seasonality_mode,
            seasonality_prior_scale=float(self.seasonality_prior_scale),
            holidays_prior_scale=float(self.holidays_prior_scale),
            changepoint_prior_scale=float(self.changepoint_prior_scale),
            mcmc_samples=self.mcmc_samples,
            interval_width=1 - self.alpha,
            uncertainty_samples=self.uncertainty_samples,
            stan_backend=self.stan_backend,
        )
        return self

    def _convert_int_to_date(self, y):
        """Convert int to date, for use by prophet."""
        y = y.copy()
        idx_max = y.index[-1] + 1
        int_idx = pd.date_range(start="2000-01-01", periods=idx_max, freq="D")
        int_idx = int_idx[y.index]
        y.index = int_idx
        return y

    def _convert_input_to_date(self, y):
        """Coerce y.index to pd.DatetimeIndex, for use by prophet."""
        if y is None:
            return None
        elif isinstance(y.index, pd.PeriodIndex):
            y = y.copy()
            y.index = y.index.to_timestamp()
        elif pd.api.types.is_integer_dtype(y.index):
            y = self._convert_int_to_date(y)
        # else y is pd.DatetimeIndex as prophet expects, and needs no conversion
        return y

    def _fit(self, y, X, fh):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.

        Returns
        -------
        self : returns an instance of self.
        """
        self._instantiate_model()

        # Remember y's input type before conversion
        self.y_index_was_period_ = isinstance(y.index, pd.PeriodIndex)
        self.y_index_was_int_ = pd.api.types.is_integer_dtype(y.index)

        # various type input indices are converted to datetime
        # since facebook prophet can only deal with dates
        y = self._convert_input_to_date(y)
        X = self._convert_input_to_date(X)

        # We have to bring the data into the required format for fbprophet
        # the index should not be pandas index, but in a column named "ds"
        df = y.copy()
        df.columns = ["y"]
        df.index.name = "ds"
        df = df.reset_index()

        # Add seasonality/seasonalities
        if self.add_seasonality:
            if isinstance(self.add_seasonality, dict):
                self._forecaster.add_seasonality(**self.add_seasonality)
            elif isinstance(self.add_seasonality, list):
                for seasonality in self.add_seasonality:
                    self._forecaster.add_seasonality(**seasonality)

        # Add country holidays
        if self.add_country_holidays:
            self._forecaster.add_country_holidays(**self.add_country_holidays)

        # Add regressor (multivariate)
        if X is not None:
            X = X.copy()
            df, X = _merge_X(df, X)
            for col in X.columns:
                self._forecaster.add_regressor(col)

        # Add floor and bottom when growth is logistic
        if self.growth == "logistic":
            if self.growth_cap is None:
                raise ValueError(
                    "Since `growth` param is set to 'logistic', expecting `growth_cap`"
                    " to be non `None`: a float."
                )

            df["cap"] = self.growth_cap
            df["floor"] = self.growth_floor

        fit_kwargs = self.fit_kwargs or {}
        if self.verbose:
            self._forecaster.fit(df=df, **fit_kwargs)
        else:
            with _suppress_stdout_stderr():
                self._forecaster.fit(df=df, **fit_kwargs)

        return self

    def _get_prophet_fh(self):
        """Get a prophet compatible fh, in datetime, even if fh was int."""
        fh = self.fh.to_absolute_index(cutoff=self.cutoff)
        if isinstance(fh, pd.PeriodIndex):
            fh = fh.to_timestamp()
        if not isinstance(fh, pd.DatetimeIndex):
            max_int = fh[-1] + 1
            fh_date = pd.date_range(start="2000-01-01", periods=max_int, freq="D")
            fh = fh_date[fh]
        return fh

    def _convert_X_for_exog(self, X, fh):
        """Conerce index of X to index expected by prophet."""
        if X is None:
            return None
        elif isinstance(X.index, pd.PeriodIndex):
            X = X.copy()
            X = X.loc[self.fh.to_absolute_index(self.cutoff)]
            X.index = X.index.to_timestamp()
        elif pd.api.types.is_integer_dtype(X.index):
            X = X.copy()
            X = X.loc[self.fh.to_absolute(self.cutoff).to_numpy()]
            X.index = fh
        # else X is pd.DatetimeIndex as prophet expects, and needs no conversion
        else:
            X = X.loc[fh]
        return X

    def _predict(self, fh=None, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : array-like
            The forecasters horizon with the steps ahead to to predict.
            Default is
            one-step ahead forecast, i.e. np.array([1]).
        X : pd.DataFrame, optional
            Exogenous data, by default None

        Returns
        -------
        y_pred : pandas.Series
            Returns series of predicted values.

        Raises
        ------
        Exception
            Error when merging data
        """
        fh = self._get_prophet_fh()
        df = pd.DataFrame({"ds": fh}, index=fh)

        X = self._convert_X_for_exog(X, fh)

        # Merge X with df (of created future DatetimeIndex values)
        if X is not None:
            df, X = _merge_X(df, X)

        if self.growth == "logistic":
            df["cap"] = self.growth_cap
            df["floor"] = self.growth_floor

        out = self._forecaster.predict(df)

        out.set_index("ds", inplace=True)
        y_pred = out.loc[:, "yhat"]

        # bring outputs into required format
        # same column names as training data, index should be index, not "ds"
        y_pred = pd.DataFrame(y_pred)
        y_pred.reset_index(inplace=True)
        y_pred.index = y_pred["ds"].values
        y_pred.drop("ds", axis=1, inplace=True)
        y_pred.columns = self._y.columns

        if self.y_index_was_int_ or self.y_index_was_period_:
            y_pred.index = self.fh.to_absolute_index(cutoff=self.cutoff)

        return y_pred

    def _predict_interval(self, fh, X, coverage):
        """Compute/return prediction quantiles for a forecast.

        private _predict_interval containing the core logic,
            called from predict_interval and possibly predict_quantiles

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon
            Forecasting horizon, default = y.index (in-sample forecast)
        X : pd.DataFrame, optional (default=None)
            Exogenous time series
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input `coverage`.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh. Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        fh = self._get_prophet_fh()

        X = self._convert_X_for_exog(X, fh)

        # prepare the return DataFrame - empty with correct cols
        var_names = self._get_varnames()
        var_name = var_names[0]

        int_idx = pd.MultiIndex.from_product([var_names, coverage, ["lower", "upper"]])
        pred_int = pd.DataFrame(columns=int_idx)

        # prepare the DataFrame to pass to prophet
        df = pd.DataFrame({"ds": fh}, index=fh)
        if X is not None:
            df, X = _merge_X(df, X)

        for c in coverage:
            # override parameters in prophet - this is fine since only called in predict
            self._forecaster.interval_width = c
            self._forecaster.uncertainty_samples = self.uncertainty_samples

            # call wrapped prophet, get prediction
            out_prophet = self._forecaster.predict(df)
            # put the index (in ds column) back in the index
            out_prophet.set_index("ds", inplace=True)
            out_prophet.index.name = None
            out_prophet = out_prophet[["yhat_lower", "yhat_upper"]]

            # retrieve lower/upper and write in pred_int return frame
            # instead of writing lower to lower, upper to upper
            #  we take the min/max for lower and upper
            #  because prophet (erroneously?) uses MC independent for upper/lower
            #  so if coverage is small, it can happen that upper < lower in prophet
            pred_int[(var_name, c, "lower")] = out_prophet.min(axis=1)
            pred_int[(var_name, c, "upper")] = out_prophet.max(axis=1)

        if self.y_index_was_int_ or self.y_index_was_period_:
            pred_int.index = self.fh.to_absolute_index(cutoff=self.cutoff)

        return pred_int

    def _get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict

        References
        ----------
        https://facebook.github.io/prophet/docs/additional_topics.html
        """
        fitted_params = {}
        for name in ["k", "m", "sigma_obs"]:
            fitted_params[name] = self._forecaster.params[name][0][0]
        for name in ["delta", "beta"]:
            fitted_params[name] = self._forecaster.params[name][0]
        return fitted_params

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict
        """
        params = {
            "n_changepoints": 0,
            "yearly_seasonality": False,
            "weekly_seasonality": False,
            "daily_seasonality": False,
            "uncertainty_samples": 10,
            "verbose": False,
            "fit_kwargs": {"seed": 12345},
        }
        return params


def _merge_X(df, X):
    """Merge X and df on the DatetimeIndex.

    Parameters
    ----------
    fh : sktime.ForecastingHorizon
    X : pd.DataFrame
        Exogeneous data
    df : pd.DataFrame
        Contains a DatetimeIndex column "ds"

    Returns
    -------
    pd.DataFrame
        DataFrame with containing X and df (with a DatetimeIndex column "ds")

    Raises
    ------
    TypeError
        Error if merging was not possible
    """
    # Merging on the index is unreliable, possibly due to loss of freq information in fh
    X.columns = X.columns.astype(str)
    if "ds" in X.columns and pd.api.types.is_numeric_dtype(X["ds"]):
        longest_column_name = max(X.columns, key=len)
        X.loc[:, str(longest_column_name) + "_"] = X.loc[:, "ds"]
        # raise ValueError("Column name 'ds' is reserved in prophet")
    X.loc[:, "ds"] = X.index
    df = df.merge(X, how="inner", on="ds", copy=True)
    X = X.drop(columns="ds")
    return df, X


class _suppress_stdout_stderr:
    """Context manager for doing  a "deep suppression" of stdout and stderr.

    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).


    References
    ----------
    https://github.com/facebook/prophet/issues/223
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
