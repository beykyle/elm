from collections import OrderedDict
from typing import Callable

import numpy as np
from scipy.stats import multivariate_normal

from exfor_tools.reaction import Reaction
from exfor_tools.distribution import AngularDistribution

import jitr
from jitr.xs.elastic import DifferentialWorkspace, ElasticXS
from jitr import rmatrix


class Constraint:
    """
    Represents an experimental constraint y, which is assumed to be
    a random variate distributed according to a multivariate_normal
    around y with an arbitrary covariance matrix

    Parameters
    ----------
    y : np.ndarray
        Experimental data output
    x : np.array
        Experimental data input
    covariance : np.ndarray
        Covariance matrix.
    """

    def __init__(self, y: np.ndarray, x: np.ndarray, covariance: np.ndarray):
        self.y = y
        if len(self.y.shape) > 1:
            raise ValueError(f"data must be 1D; y was {len(self.y.shape)}D")
        self.x = x
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
        self.dist = multivariate_normal(
            mean=self.y, cov=self.covariance, allow_singular=True
        )

    def residual(self, model, params):
        """
        Calculate the residuals.

        Parameters
        ----------
        params : array-like
            Parameters for the model.

        Returns
        -------
        np.ndarray
            Residuals.
        """
        return self.y - model(self.x, params)

    def chi2(self, model, params):
        """
        Calculate the generalised chi-squared statistic.

        Parameters
        ----------
        params : array-like
            Parameters for the model.

        Returns
        -------
        float
            Chi-squared statistic.
        """
        delta = self.residual(model, params)
        return delta.T @ self.cov_inv @ delta

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

    def probability_within_interval(self, ylow: np.ndarray, yhigh: np.ndarray):
        """
        Returns the probability that self.y falls within ylow and yhigh, for a
        generalized measure of empirical coverage

        Parameters
        ----------
        ylow : np.ndarray, same shape as self.y
        yhigh : np.ndarray, same shape as self.y

        Returns
        -------
        float
        """
        prob = self.dist.cdf(yhigh) - self.dist.cdf(ylow)
        return prob


class ReactionDistribution(Constraint):
    """
    Represents the constraint for a particular reaction observable, calculating the
    appropriate covariance matrix given statistical and systematic errors.
    """

    def __init__(
        self,
        quantity: str,
        measurement: AngularDistribution,
        normalize=None,
        include_sys_norm_err=True,
        include_sys_offset_err=True,
        include_sys_gen_err=True,
    ):
        """
        Params:
        quantity : str
            The name of the quantity being measured.
        measurement : AngularDistribution
            An object containing the angular distribution measurements along with
            their statistical and systematic errors.
        normalize : float, optional
            A value to normalize the measurements and errors. If None, no normalization
            is applied. Default is None.
        include_sys_norm_err : bool, optional
            If True, includes systematic normalization error in the covariance matrix.
            Default is True.
        include_sys_offset_err : bool, optional
            If True, includes systematic offset error in the covariance matrix.
            Default is True.
        include_sys_gen_err : bool, optional
            If True, includes general systematic error in the covariance matrix.
            Default is True.

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
        super().__init__(y, x, covariance)
        self.stat_err_y = stat_err_y
        self.sys_err_norm = sys_err_norm
        self.sys_err_offset = sys_err_offset
        self.sys_err_general = sys_err_general


def check_angle_grid(angles_rad: np.ndarray, name: str):
    if len(angles_rad.shape) > 1:
        raise ValueError(f"{name} must be 1D, is {len(angles_rad.shape)}D")
    if angles_rad[0] < 0 or angles_rad[-1] > np.pi:
        raise ValueError(f"{name} must be on [0,pi)")


class ElasticModel:
    """
    Encapsulates any reaction model for differential elastic quantities using a
    jitr.xs.elastic.DifferentialWorkspace.
    """

    def __init__(
        self,
        quantity: str,
        reaction: Reaction,
        Elab: float,
        angles_rad_vis: np.ndarray,
        angles_rad_constraint: np.ndarray,
        core_solver: rmatrix.Solver,
        lmax: int = 20,
    ):
        """
        Params:
            quantity: The type of quantity to be calculated (e.g., "dXS/dA",
                "dXS/dRuth", "Ay").
            reaction: The reaction object containing details of the reaction.
            Elab: The laboratory energy.
            angles_rad_vis: Array of angles in radians for visualization.
            angles_rad_constraint: Array of angles in radians corresponding to
                experimentally measured constraints.
            core_solver: The core solver used for calculations.
            lmax: Maximum angular momentum, defaults to 20.
        """
        self.quantity = quantity
        self.reaction = reaction

        check_angle_grid(angles_rad_vis, "angles_rad_vis")
        check_angle_grid(angles_rad_constraint, "angles_rad_constraint")

        constraint_ws, visualization_ws, kinematics = set_up_solver(
            reaction,
            Elab,
            angles_rad_constraint,
            angles_rad_vis,
            core_solver,
            lmax,
        )
        self.constraint_workspace = constraint_ws
        self.visualization_workspace = visualization_ws
        self.kinematics = kinematics
        self.Elab = Elab

        if self.quantity == "dXS/dA":
            self.get_quantity = self.get_diff_xs
            self.get_quantity_vis = self.get_diff_xs_vis
        elif self.quantity == "dXS/dRuth":
            self.get_quantity = self.get_diff_xs_ratio_Rutherford
            self.get_quantity_vis = self.get_diff_xs_ratio_Rutherford_vis
        elif self.quantity == "Ay":
            self.get_quantity = self.get_Ay
            self.get_quantity_vis = self.get_Ay_vis

    def __call__(
        self,
        model: Callable[[DifferentialWorkspace, OrderedDict], ElasticXS],
        params: OrderedDict,
    ) -> np.ndarray:
        """
        Params:
            model: The model function to be used for calculations.
            params: Parameters for the model.
        Returns:
            Calculated quantity as a numpy array.
        """
        return self.get_quantity(model, params)

    def get_xs_constraint(
        self,
        model: Callable[[DifferentialWorkspace, OrderedDict], ElasticXS],
        params: OrderedDict,
    ) -> ElasticXS:
        """
        Params:
            model: The model function to be used for calculations.
            params: Parameters for the model.
        Returns:
            Elastic cross-section.
        """
        return model(self.constraint_workspace, params)

    def get_xs_vis(
        self,
        model: Callable[[DifferentialWorkspace, OrderedDict], ElasticXS],
        params: OrderedDict,
    ) -> ElasticXS:
        """
        Params:
            model: The model function to be used for calculations.
            params: Parameters for the model.
        Returns:
            Elastic cross-section.
        """
        return model(self.visualization_workspace, params)

    def get_diff_xs(
        self,
        model: Callable[[DifferentialWorkspace, OrderedDict], ElasticXS],
        params: OrderedDict,
    ) -> np.ndarray:
        """
        Params:
            model: The model function to be used for calculations.
            params: Parameters for the model.
        Returns:
            Differential cross-section as a numpy array.
        """
        return self.get_xs_constraint(model, params).dsdo

    def get_diff_xs_vis(
        self,
        model: Callable[[DifferentialWorkspace, OrderedDict], ElasticXS],
        params: OrderedDict,
    ) -> np.ndarray:
        """
        Params:
            model: The model function to be used for calculations.
            params: Parameters for the model.
        Returns:
            Differential cross-section as a numpy array.
        """
        return self.get_xs_vis(model, params).dsdo

    def get_Ay(
        self,
        model: Callable[[DifferentialWorkspace, OrderedDict], ElasticXS],
        params: OrderedDict,
    ) -> np.ndarray:
        """
        Params:
            model: The model function to be used for calculations.
            params: Parameters for the model.
        Returns:
            Analyzing power Ay as a numpy array.
        """
        return self.get_xs_constraint(model, params).Ay

    def get_Ay_vis(
        self,
        model: Callable[[DifferentialWorkspace, OrderedDict], ElasticXS],
        params: OrderedDict,
    ) -> np.ndarray:
        """
        Params:
            model: The model function to be used for calculations.
            params: Parameters for the model.
        Returns:
            Analyzing power Ay as a numpy array.
        """
        return self.get_xs_vis(model, params).Ay

    def get_diff_xs_ratio_Rutherford(
        self,
        model: Callable[[DifferentialWorkspace, OrderedDict], ElasticXS],
        params: OrderedDict,
    ) -> np.ndarray:
        """
        Params:
            model: The model function to be used for calculations.
            params: Parameters for the model.
        Returns:
            Ratio as a numpy array.
        """
        return (
            self.get_xs_constraint(model, params).dsdo
            / self.constraint_workspace.rutherford
        )

    def get_diff_xs_ratio_Rutherford_vis(
        self,
        model: Callable[[DifferentialWorkspace, OrderedDict], ElasticXS],
        params: OrderedDict,
    ) -> np.ndarray:
        """
        Params:
            model: The model function to be used for calculations.
            params: Parameters for the model.
        Returns:
            Ratio as a numpy array.
        """
        return (
            self.get_xs_vis(model, params).dsdo
            / self.visualization_workspace.rutherford
        )


def set_up_solver(
    reaction: Reaction,
    Elab: float,
    angle_rad_constraint: np.array,
    angle_rad_vis: np.array,
    core_solver: rmatrix.Solver,
    lmax: int,
):
    """
    Set up the solver for the reaction.

    Parameters
    ----------
    reaction : exfor_tools.Reaction
        Reaction information.
    Elab : float
        Laboratory energy.
    angle_rad_constraint : np.array
        Angles to compare to experiment (rad).
    angle_rad_vis : np.array
        Angles to visualize on (rad)
    core_solver : jitr.rmatrix.Solver
        Core solver.
    lmax : int
        Maximum angular momentum.

    Returns
    -------
    tuple
        constraint and visualization workspaces.
    """

    # get kinematics and parameters for this experiment
    kinematics = reaction.kinematics(Elab)
    interaction_range_fm = jitr.utils.interaction_range(reaction.projectile.A)
    a = interaction_range_fm * kinematics.k + 2 * np.pi
    channel_radius_fm = a / kinematics.k
    Ns = jitr.utils.suggested_basis_size(a)
    N = core_solver.kernel.quadrature.nbasis
    if Ns > N:
        raise ValueError(
            f"Suggested basis size for dimensionless channel radius {a} "
            f"is {Ns}, but core_solver only has {N}"
        )

    integral_ws = jitr.xs.elastic.IntegralWorkspace(
        reaction=reaction,
        kinematics=kinematics,
        channel_radius_fm=channel_radius_fm,
        solver=core_solver,
        lmax=lmax,
    )

    constraint_ws = jitr.xs.elastic.DifferentialWorkspace(
        integral_workspace=integral_ws, angles=angle_rad_constraint
    )
    visualization_ws = jitr.xs.elastic.DifferentialWorkspace(
        integral_workspace=integral_ws, angles=angle_rad_vis
    )

    return constraint_ws, visualization_ws, kinematics
