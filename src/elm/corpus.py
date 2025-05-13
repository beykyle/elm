from collections import OrderedDict
import numpy as np
from exfor_tools.distribution import AngularDistribution
from exfor_tools.reaction import Reaction
from .constraint import Constraint, ReactionConstraint
from .model import ElasticModel


class Corpus:
    """
    A class to represent a collection of constraints.

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
    def __init__(self, constraints: list[Constraint]):
        self.constraints = constraints
        self.y = np.hstack([constraint.y for constraint in self.constraints])
        self.x = np.hstack([constraint.x for constraint in self.constraints])
        self.n_data_pts = self.y.size

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
        Compute the chi-squared value for the given parameters.

        Parameters
        ----------
        params : OrderedDict
            Parameters for which to compute the chi-squared value.

        Returns
        -------
        float
            Chi-squared value for the given parameters.
        """
        return sum(constraint.chi2(params) for constraint in self.constraints)

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
            return sum(
                constraint.num_pts_within_interval(ylow, yhigh) / constraint.n_data_pts
                for constraint in self.constraints
            )
        elif method == "average":
            return sum(
                constraint.expected_num_pts_within_interval(ylow, yhigh)
                / constraint.n_data_pts
                for constraint in self.constraints
            )


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
        quantity: str,
        measurements: list[AngularDistribution],
        reactions: list[Reaction],
        angles_vis=None,
        constraint_kwargs=None,
        lmax=30,
    ):
        """
        Initialize an ElasticAngularCorpus instance from a list of measured
        AngularDistributions and corresponding Reactions

        Parameters
        ----------
        quantity : str
            The quantity to be measured.
        measurements : list of AngularDistribution
            A list of angular distribution measurements.
        reactions : list of Reaction
            A list of reactions.
        angles_vis : list, optional
            Angles for visualization. Defaults to None.
        constraint_kwargs : dict, optional
            Additional keyword arguments for constraints. Defaults to None.
        lmax : int, optional
            Maximum angular momentum. Defaults to 30.
        """
        self.quantity = quantity
        self.angles_vis = (
            angles_vis if angles_vis is not None else np.linspace(0.01, 180, 90)
        )
        self.lmax = lmax
        constraint_kwargs = constraint_kwargs or {}
        constraints = []
        for measurement, reaction in zip(measurements, reactions):

            model = ElasticModel(
                quantity=self.quantity,
                reaction=reaction,
                Elab=measurement.Einc,
                angles_rad_constraint=measurement.x * np.pi / 180,
                angles_rad_vis=angles_vis * np.pi / 180,
                lmax=30,
            )

            if self.quantity == "dXS/dRuth" and measurement.quantity == "dXS/dA":
                norm = model.constraint_workspace.rutherford
            else:
                norm = None

            constraints.append(
                ReactionConstraint(
                    self.quantity,
                    measurement,
                    model,
                    normalize=norm,
                    **constraint_kwargs,
                )
            )
        super().__init__(constraints)
