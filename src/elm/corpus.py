from collections import OrderedDict
from typing import Callable

import numpy as np

from exfor_tools.distribution import AngularDistribution
from exfor_tools.reaction import Reaction

from jitr.xs.elastic import DifferentialWorkspace, ElasticXS

from .constraint import Constraint, ReactionConstraint
from .model import ElasticModel, ElasticWorkspace


class Corpus:
    """
    A class to represent a generic collection of constraints.

    Attributes
    ----------
    constraints : list of Constraint
        A list of constraints.
    y : np.ndarray
        Combined y values from all constraints.
    x : np.ndarray
        Combined x values from all constraints.
    n_data_pts : int
        Total number of data points.
    nparams : int
        Total number of free parameters in the model.
    model_name : str
        name or label for the model

    Methods
    -------
    residual(params)
        Computes the residuals for the given parameters.
    chi2(params)
        Computes the chi-squared value for the given parameters.
    empirical_coverage(ylow, yhigh, method='count')
        Computes the empirical coverage within the given interval.
    """

    def __init__(
        self,
        constraints: list[Constraint],
        n_params: int,
        model_name: str,
        weights: np.ndarray = None,
    ):
        self.constraints = constraints
        self.n_params = n_params
        self.model_name = model_name
        self.y = np.hstack([constraint.y for constraint in self.constraints])
        self.x = np.hstack([constraint.x for constraint in self.constraints])
        self.n_data_pts = self.y.size
        self.n_dof = self.n_data_pts - self.n_params
        if self.n_dof < 0:
            raise ValueError(
                f"Model under-constrained! {self.n_params} free parameters and {self.n_data_pts} data points"
            )

        if weights is None:
            weights = np.ones((len(self.constraints),), dtype=float)
        elif weights.shape != (len(self.constraints),):
            raise ValueError(
                "weights must be a 1D array with the same shape as constraints"
            )
        self.weights = weights

    def residual(self, params: OrderedDict):
        """
        Compute the residuals for the given parameters.

        Parameters
        ----------
        params : OrderedDict
            Parameters for which to compute the residuals.

        Returns
        -------
        np.ndarray
            Residuals for the given parameters.
        """
        return np.hstack(
            [constraint.residual(params) for constraint in self.constraints]
        )

    def chi2(self, params: OrderedDict):
        """
        Compute the weighted chi-squared value for the given parameters.

        Parameters
        ----------
        params : OrderedDict
            Parameters for which to compute the chi-squared value.

        Returns
        -------
        float
            Chi-squared value for the given parameters.
        """
        return sum(
            constraint.chi2(params)
            for weight, constraint in zip(self.weights, self.constraints)
        )

    def empirical_coverage(
        self, ylow: np.ndarray, yhigh: np.ndarray, method: str = "count"
    ):
        """
        Compute the empirical coverage within the given interval.

        Parameters
        ----------
        ylow : np.ndarray
            Lower bounds of the interval.
        yhigh : np.ndarray
            Upper bounds of the interval.
        method : str, optional
            Method to compute coverage ('count' or 'average'), by
            default 'count'.

        Returns
        -------
        float
            Empirical coverage within the given interval.
        """
        if method == "count":
            return (
                sum(
                    constraint.num_pts_within_interval(ylow, yhigh)
                    for constraint in self.constraints
                )
                / self.n_data_pts
            )
        elif method == "average":
            return (
                sum(
                    constraint.expected_num_pts_within_interval(ylow, yhigh)
                    for constraint in self.constraints
                )
                / self.n_data_pts
            )


def build_workspaces_from_measurements(
    quantity: str,
    measurements: list[tuple[Reaction, AngularDistribution]],
    angles_vis=None,
    lmax=30,
):
    angles_vis = angles_vis if angles_vis is not None else np.linspace(0.01, 180, 90)
    workspaces = []
    for reaction, measurement in measurements:
        workspace = ElasticWorkspace(
            quantity=quantity,
            reaction=reaction,
            Elab=measurement.Einc,
            angles_rad_constraint=measurement.x * np.pi / 180,
            angles_rad_vis=angles_vis * np.pi / 180,
            lmax=lmax,
        )
        workspaces.append(workspace)
    return workspaces


class ElasticAngularCorpus(Corpus):
    """
    A class to represent a collection of elastic angular constraints.

    Attributes
    ----------
    quantity : str
        The quantity being measured.
    angles_vis : np.ndarray
        Angles for visualization.
    lmax : int
        Maximum angular momentum.
    constraints : list of ReactionConstraint
        A list of reaction constraints.
    """

    def __init__(
        self,
        model: Callable[[DifferentialWorkspace, OrderedDict], ElasticXS],
        model_name: str,
        n_params: int,
        quantity: str,
        workspaces: list[ElasticWorkspace],
        measurements: list[tuple[Reaction, AngularDistribution]],
        weights=None,
        **constraint_kwargs,
    ):
        """
        Initialize an `ElasticAngularCorpus` instance from a list of measured
        `AngularDistribution`s and corresponding `Reaction`s, along with the
        corresponding `ElasticWorkspace`s

        Parameters
        ----------
        quantity : str
            The quantity to be measured.
        nparams : int
            Total number of free parameters in the model.
        workspaces : list[tuple[AngularDistribution, ElasticWorkspace]]
            A list of tuples containing a reaction and corresponding measured
            angular distribution of given quantity.
        constraint_kwargs : dict, optional
            Additional keyword arguments for constraints. Defaults to `None`.
        """
        constraint_kwargs = constraint_kwargs or {}
        constraints = []

        self.quantity = quantity
        self.measurements = measurements

        for (reaction, measurement), workspace in zip(measurements, workspaces):
            if (
                workspace.kinematics.Elab != measurement.Einc
                or workspace.reaction != reaction
                or not np.allclose(
                    workspace.constraint_workspace.angles, measurement.x * np.pi / 180
                )
            ):
                raise ValueError(
                    f"mismatch between workspace and measurement for subentry {measurement.subentry}."
                    "\n          workspace | measuremnt  "
                    "\n ======================================================"
                    f"\n Energy:      {workspace.kinematics.Elab} | {measurement.Einc}"
                    f"\n Reacttion:   {workspace.reaction} |  {reaction}"
                    f"\n Same angles: {np.allclose(workspace.constraint_workspace.angles, measurement.x * np.pi / 180)})"
                )
            if self.quantity == "dXS/dRuth" and measurement.quantity == "dXS/dA":
                norm = workspace.constraint_workspace.rutherford / 1000
            elif self.quantity == "dXS/dA" and measurement.quantity == "dXS/dRuth":
                norm = 1000.0 / workspace.constraint_workspace.rutherford
            else:
                norm = None
            constraints.append(
                ReactionConstraint(
                    quantity=self.quantity,
                    measurement=measurement,
                    model=ElasticModel(workspace, model),
                    normalize=norm,
                    **constraint_kwargs,
                )
            )
        super().__init__(constraints, n_params, model_name, weights)

    def set_model(
        self,
        model: Callable[[DifferentialWorkspace, OrderedDict], ElasticXS],
        **constraint_kwargs,
    ):
        """
        keeping the same measurements (and their corresponding workspaces), reset
        constraints to use a new model
        """
        for i in range(len(self.constraints)):
            self.constraints[i] = ReactionConstraint(
                quantity=self.quantity,
                measurement=self.measurements[i],
                model=ElasticModel(self.constraints[i].workspace, model),
                **constraint_kwargs,
            )
