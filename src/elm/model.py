from collections import OrderedDict
from typing import Callable

import numpy as np

from exfor_tools.reaction import Reaction

import jitr
from jitr.xs.elastic import DifferentialWorkspace, ElasticXS
from jitr import rmatrix


class Model:
    """
    Represents an arbitrary parameteric model
    """

    def __init__(self, x: np.ndarray):
        self.x = x

    def __call__(self, params: OrderedDict):
        pass


class ElasticModel(Model):
    """
    Encapsulates any reaction model for differential elastic quantities using a
    jitr.xs.elastic.DifferentialWorkspace.
    """

    def __init__(
        self,
        model: Callable[[DifferentialWorkspace, OrderedDict], ElasticXS],
        quantity: str,
        reaction: Reaction,
        Elab: float,
        angles_rad_vis: np.ndarray,
        angles_rad_constraint: np.ndarray,
        lmax: int = 20,
    ):
        """
        Params:
            model: A callable that takes in a DifferentialWorkspace and an
                OrderedDict of params and spits out the corresponding ElasticXS
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

        super().__init__(angles_rad_constraint)

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
        self.model = model
        self.quantity_extractor = get_quantity_extractor(self.quantity)

        super().__init__(self.model.x)

    def get_model_xs(self, params: OrderedDict) -> ElasticXS:
        return self.model(self.constraint_workspace, params)

    def get_model_xs_visualization(self, params: OrderedDict) -> ElasticXS:
        return self.model(self.visualization_workspace, params)

    def __call__(self, params: OrderedDict) -> np.ndarray:
        """
        Params:
            model: The model function to be used for calculations.
            params: Parameters for the model.
        Returns:
            Calculated quantity as a numpy array (same shape as
                angle_rad_constraint).
        """
        xs = self.get_model_xs(params)
        return self.quantity_extractor(xs, self.constraint_workspace)


def get_quantity_extractor(quantity: str):
    if quantity == "dXS/dA":
        return lambda xs, ws: xs.dsdo
    elif quantity == "dXS/dRuth":
        return lambda xs, ws: xs.dsdo / ws.rutherford
    elif quantity == "Ay":
        return lambda xs, ws: xs.Ay
    else:
        raise ValueError(f"Unknown quantity {quantity}")


def set_up_solver(
    reaction: Reaction,
    Elab: float,
    angle_rad_constraint: np.array,
    angle_rad_vis: np.array,
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
    core_solver = rmatrix.Solver(Ns)

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


def check_angle_grid(angles_rad: np.ndarray, name: str):
    if len(angles_rad.shape) > 1:
        raise ValueError(f"{name} must be 1D, is {len(angles_rad.shape)}D")
    if angles_rad[0] < 0 or angles_rad[-1] > np.pi:
        raise ValueError(f"{name} must be on [0,pi)")
