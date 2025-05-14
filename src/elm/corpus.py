from collections import OrderedDict
from typing import Callable

import numpy as np

from exfor_tools.curate import MulltiQuantityReactionData
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

    Methods
    -------
    residual(params)
        Computes the residuals for the given parameters.
    chi2(params)
        Computes the chi-squared value for the given parameters.
    empirical_coverage(ylow, yhigh, method='count')
        Computes the empirical coverage within the given interval.
    """

    def __init__(self, constraints: list[Constraint], weights: np.ndarray = None):
        self.constraints = constraints
        self.y = np.hstack([constraint.y for constraint in self.constraints])
        self.x = np.hstack([constraint.x for constraint in self.constraints])
        self.n_data_pts = self.y.size

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


def build_workspaces_from_data(
    quantity: str,
    data: dict[tuple[int, int], MulltiQuantityReactionData],
    angles_vis=None,
    lmax=30,
):
    angles_vis = angles_vis if angles_vis is not None else np.linspace(0.01, 180, 90)
    measurements = []
    for target, data_set in data.items():
        for entry_id, entry in data_set.data[quantity].entries.items():
            for measurement in entry.measurements:
                measurements.append((entry.reaction, measurement))

    # sort by energy
    # measurements.sort(key=lambda m: m[1].Einc, reverse=True)

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
        workspaces.append((measurement, workspace))
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
        quantity: str,
        workspaces: list[tuple[AngularDistribution, ElasticWorkspace]],
        weights=None,
        constraint_kwargs=None,
    ):
        """
        Initialize an `ElasticAngularCorpus` instance from a list of measured
        `AngularDistribution`s and corresponding `Reaction`s, along with the
        corresponding `ElasticWorkspace`s

        Parameters
        ----------
        quantity : str
            The quantity to be measured.
        workspaces : list[tuple[AngularDistribution, ElasticWorkspace]]
            A list of tuples containing a reaction and corresponding measured
            angular distribution of given quantity.
        constraint_kwargs : dict, optional
            Additional keyword arguments for constraints. Defaults to `None`.
        """
        constraint_kwargs = constraint_kwargs or {}
        constraints = []

        self.quantity = quantity

        for measurement, workspace in workspaces:
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
        super().__init__(constraints, weights)
