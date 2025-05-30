from collections import OrderedDict

import numpy as np

# from scipy.stats import multivariate_normal

from exfor_tools.distribution import Distribution
from .model import Model


def log_likelihood_cholesky(y, mu, cov):
    """
    Will need this if cov_inv can't be precomputed
    """
    n = len(y)
    delta = y - mu

    # Cholesky decomposition: cov = L @ L.T
    L = np.linalg.cholesky(cov, lower=True)

    # Solve L z = delta --> z = L^{-1} delta
    z = np.lonalg.solve_triangular(L, delta, lower=True)

    # Mahalanobis distance: z^T z
    mahalanobis = np.dot(z, z)

    # Log determinant: log(det(Sigma)) = 2 * sum(log(diag(L)))
    log_det = 2 * np.sum(np.log(np.diag(L)))

    return -0.5 * (mahalanobis + log_det + n * np.log(2 * np.pi))


class Constraint:
    """
    Represents experimental data y, which is assumed to be
    a random variate distributed according to a multivariate normal
    around y with an arbitrary covariance matrix, along with some
    parameteric Model which produces predictions for y given a
    set of parameters

    Parameters
    ----------
    y : np.ndarray
        Experimental data output
    covariance : np.ndarray
        Covariance matrix.
    model: Model
        The parametric model. Must be a callable which takes in an
        OrderedDict of parameters and outputs a np.ndarray of the
        same shape as y
    """

    def __init__(self, y: np.ndarray, covariance: np.ndarray, model: Model):
        self.y = y
        if len(self.y.shape) > 1:
            raise ValueError(f"data must be 1D; y was {len(self.y.shape)}D")
        self.model = model
        self.x = model.x
        if self.x.shape != self.y.shape:
            raise ValueError(
                "Incompatible x and y shapes: " f"{self.x.shape} and {self.y.shape}"
            )
        self.n_data_pts = y.shape[0]
        if covariance.shape == (self.n_data_pts,):
            self.covariance = np.diag(covariance)
        elif covariance.shape == (self.n_data_pts, self.n_data_pts):
            self.covariance = covariance
        else:
            raise ValueError(
                f"Incompatible covariance matrix shape "
                f"{covariance.shape} for Constraint with "
                f"{self.n_data_pts} data points"
            )

        self.cov_inv = np.linalg.inv(self.covariance)
        sign, self.log_det = np.linalg.slogdet(self.covariance)
        if sign != +1:
            raise ValueError("Invalid covariance matrix! Must be positive definite.")

        # self.y_distribution = multivariate_normal(
        #    mean=self.y,
        #    cov=self.covariance,
        #    allow_singular=False,
        # )

    def residual(self, params: OrderedDict):
        """
        Calculate the residuals.

        Parameters
        ----------
        params : OrderedDict
            parameters of model

        Returns
        -------
        np.ndarray
            Residuals.
        """
        ym = self.model(params)
        return self.y - ym

    def chi2(self, params: OrderedDict):
        """
        Calculate the generalised chi-squared statistic. This is the
        Malahanobis distance between y and model(params).

        Parameters
        ----------
        params : OrderedDict
            parameters of model

        Returns
        -------
        float
            Chi-squared statistic.
        """
        delta = self.residual(params)
        return delta.T @ self.cov_inv @ delta

    def logpdf(self, params: OrderedDict):
        """
        Returns the logpdf that the Model, given params, reproduces y

        Parameters
        ----------
        params : OrderedDict
            parameters of model

        Returns
        -------
        float
        """

        # ym = self.model(params)
        # return self.y_distribution.logpdf(ym)
        r = self.residual(params)
        mahalanobis = r.T @ self.cov_inv @ r
        return -0.5 * (mahalanobis + self.log_det + self.n_data_pts * np.log(2 * np.pi))

    def num_pts_within_interval(self, ylow: np.ndarray, yhigh: np.ndarray):
        """
        Returns the number of points in y that fall between ylow and yhigh,
        useful for calculating emperical coverages

        Parameters
        ----------
        ylow : np.ndarray, same shape as self.y
        yhigh : np.ndarray, same shape as self.y

        Returns
        -------
        int
        """
        return int(np.sum(np.logical_and(self.y >= ylow, self.y < yhigh)))

    def expected_num_pts_within_interval(self, ylow: np.ndarray, yhigh: np.ndarray):
        """
        Returns the number of points multiplied by the probability that self.y
        falls within ylow and yhigh, for a generalized measure of empirical
        coverage. This is the expectation value of the number of points in
        y that fall between ylow and yhigh

        Parameters
        ----------
        ylow : np.ndarray, same shape as self.y
        yhigh : np.ndarray, same shape as self.y

        Returns
        -------
        float
        """
        pass
        # prob = self.y_distribution.cdf(yhigh) - self.y_distribution.cdf(ylow)
        # return prob * self.n_data_pts


class ReactionConstraint(Constraint):
    """
    Represents the constraint determined by a AngularDistribution,
    with the appropriate covariance matrix given statistical
    and systematic errors.
    """

    def __init__(
        self,
        quantity: str,
        measurement: Distribution,
        model: Model,
        normalize=None,
        include_sys_norm_err=True,
        include_sys_offset_err=True,
        include_sys_gen_err=True,
    ):
        """
        Params:
        quantity : str
            The name of the measured quantity
        measurement : Distribution
            An object containing the measured values along
            with their statistical and systematic errors.
        model : ElasticModel
            The model to predict the quantity
        normalize : float, optional
            A value to normalize the measurements and errors. If None, no
            normalization is applied. Default is None.
        include_sys_norm_err : bool, optional
            If True, includes systematic normalization error in the covariance
            matrix. Default is True.
        include_sys_offset_err : bool, optional
            If True, includes systematic offset error in the covariance matrix.
            Default is True.
        include_sys_gen_err : bool, optional
            If True, includes general systematic error in the covariance
            matrix. Default is True.

        """
        self.quantity = quantity
        self.subentry = measurement.subentry
        x = np.copy(measurement.x)
        y = np.copy(measurement.y)
        stat_err_y = np.copy(measurement.statistical_err)
        sys_err_norm = np.copy(measurement.systematic_norm_err)
        sys_err_offset = np.copy(measurement.systematic_offset_err)
        sys_err_general = np.copy(measurement.general_systematic_err)

        if normalize is not None:
            y /= normalize
            stat_err_y /= normalize
            sys_err_general /= normalize
            if sys_err_offset > 0:
                sys_err_general += sys_err_offset * np.ones_like(x) / normalize
                sys_err_offset = 0

        covariance = np.diag(stat_err_y**2)
        if include_sys_norm_err:
            covariance += np.outer(y, y) * sys_err_norm**2
        if include_sys_offset_err:
            n = y.shape[0]
            covariance += np.ones((n, n)) * sys_err_offset
        if include_sys_gen_err:
            covariance += np.outer(sys_err_general, sys_err_general)

        self.stat_err_y = stat_err_y
        self.sys_err_norm = sys_err_norm
        self.sys_err_offset = sys_err_offset
        self.sys_err_general = sys_err_general

        super().__init__(y, covariance, model)
