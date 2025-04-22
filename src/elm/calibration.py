from collections import OrderedDict

import numpy as np

import exfor_tools
import jitr

from .model import calculate_parameters, central_plus_coulomb, spin_orbit


# TODO make this model agnostic, e.g. pass in model callable
class Constraint:
    """
    Represents an experimental constraint y, with a covariance matrix,
    and some model for y, f, that takes in some params.

    Parameters
    ----------
    y : np.ndarray
        Experimental data.
    covariance : np.ndarray
        Covariance matrix.
    """

    def __init__(self, y: np.ndarray, covariance: np.ndarray):
        self.y = y
        if len(self.y.shape) > 1:
            raise ValueError(f"data must be 1D; y was {len(self.y.shape)}D")
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

    def f(self, params):
        """
        Model function for y.

        Parameters
        ----------
        params : array-like
            Parameters for the model.
        """
        pass

    def residual(self, params):
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
        return self.y - self.f(params)

    def chi2(self, params):
        """
        Calculate the chi-squared statistic.

        Parameters
        ----------
        params : array-like
            Parameters for the model.

        Returns
        -------
        float
            Chi-squared statistic.
        """
        delta = self.residual(params)
        return delta.T @ self.cov_inv @ delta


class ChExIASConstraint(Constraint):
    """
    Placeholder for ChExIASConstraint.
    """

    pass  # TODO


# TODO separate Constraint from solver
class DifferentialElasticConstraint(Constraint):
    """
    Represents a differential elastic constraint for an arbitrary model
    that provides callables to jitr.xs.elastic.DifferentialWorkspace

    Parameters
    ----------
    reaction : exfor_tools.reaction.Reaction
        Reaction information.
    quantity : str
        Quantity type.
    measurement :
        exfor_tools.angular_distribution.AngularDistributionSysStatErr
        Measurement data.
    angles_vis : np.array
        Visualization angles in degrees.
    core_solver : jitr.rmatrix.Solver
        Core solver.
    lmax : int, optional
        Maximum angular momentum, by default 20.
    divide_by_Rutherford : bool, optional
        Absolute cross-section flag, by default True.
    """

    def __init__(
        self,
        reaction: exfor_tools.reaction.Reaction,
        quantity: str,
        measurement: exfor_tools.angular_distribution.AngularDistributionSysStatErr,
        angles_vis: np.array,
        core_solver: jitr.rmatrix.Solver,
        lmax: int = 20,
        divide_by_Rutherford=False,
        workspaces=None,
    ):
        self.reaction = reaction
        self.quantity = quantity
        self.measurement = measurement

        if self.measurement.y_units == "barns/ster":
            self.measurement.y /= 1000
            self.measurement.statistical_err /= 1000
            self.measurement.systematic_offset_err /= 1000
            self.measurement.y_units = "mb/Sr"
            assert self.quantity == "dXS/dA"
        elif self.measurement.y_units == "mb/Sr":
            assert self.quantity == "dXS/dA"
        elif self.measurement.y_units == "no-dim":
            assert self.quantity == "Ay" or self.quantity == "dXS/dRuth"
        else:
            raise ValueError(f"invalid y units {self.measurement.y_units}")

        if self.measurement.x_units != "degrees":
            raise ValueError(f"invalid x units {self.measurement.x_units}")
        self.x_vis = angles_vis * np.pi / 180
        self.x_cal = self.measurement.x * np.pi / 180

        covariance = np.diag(self.measurement.statistical_err**2)
        if self.measurement.systematic_norm_err > 0:
            covariance += (
                np.outer(self.measurement.y, self.measurement.y)
                * self.measurement.systematic_norm_err**2
            )
        if self.measurement.systematic_offset_err > 0:
            raise NotImplementedError(
                "Systematic offset error covariance not implemented"
            )
        if self.measurement.general_systematic_err > 0:
            raise NotImplementedError(
                "Nonuniform systematic error covariance not implemented"
            )
        super().__init__(self.measurement.y, covariance)

        if workspaces is not None:
            calibrator, visualizer = workspaces
        else:
            calibrator, visualizer = set_up_solver(
                self.reaction,
                self.measurement.Einc,
                self.x_cal,
                self.x_vis,
                core_solver,
                lmax,
            )
        self.calibration_workspace = calibrator
        self.visualization_workspace = visualizer

        self.Elab = self.measurement.Einc
        self.mu = self.calibration_workspace.mu
        self.Ecm = self.calibration_workspace.Ecm
        self.k = self.calibration_workspace.k
        self.eta = self.calibration_workspace.eta

        if divide_by_Rutherford:
            if not self.quantity == "dXS/dRuth":
                raise ValueError(f"Why are you dividing by Rutherford? Quantity is {self.quantity}")
            self.exp.data[2, :] /= self.calibration_workspace.rutherford
            self.exp.data[3, :] /= self.calibration_workspace.rutherford

        if self.quantity == "dXS/dA":
            self.f = self.get_diff_xs
            self.f_vis = self.get_diff_xs_vis
        elif self.quantity == "dXS/dRuth":
            if self.reaction.projectile.Z == 0:
                raise ValueError("Can't do dXS/dRuth for uncharged projectile")
            self.f = self.get_diff_xs_ratio_to_ruth
            self.f_vis = self.get_diff_xs_ratio_to_ruth_vis
        elif self.quantity == "Ay":
            self.f = self.get_Ay
            self.f_vis = self.get_Ay_vis
        else:
            raise NotImplementedError(f"{self.quantity} not supported")


    def xs_cal(self, sub_params, model):
        """
        Calculate cross-section for calibration.

        Parameters
        ----------
        sub_params : OrderedDict
            Sub-parameters for the model.

        Returns
        -------
        object
            Cross-section result.
        """
        return model(sub_params, self.calibration_workspace, self.reaction)

    def xs_vis(self, sub_params, model):
        """
        Calculate cross-section for visualization.

        Parameters
        ----------
        sub_params : OrderedDict
            Sub-parameters for the model.

        Returns
        -------
        object
            Cross-section result.
        """
        return model(sub_params, self.visualization_workspace, self.reaction)

    def get_Ay(self, sub_params, model):
        """
        Get analyzing power Ay.

        Parameters
        ----------
        sub_params : OrderedDict
            Sub-parameters for the model.

        Returns
        -------
        object
            Analyzing power Ay.
        """
        return self.xs_cal(sub_params, model).Ay

    def get_Ay_vis(self, sub_params, model):
        """
        Get analyzing power Ay for visualization.

        Parameters
        ----------
        sub_params : OrderedDict
            Sub-parameters for the model.

        Returns
        -------
        object
            Analyzing power Ay.
        """
        return self.xs_vis(sub_params, model).Ay

    def get_diff_xs(self, sub_params, model):
        """
        Get differential cross-section.

        Parameters
        ----------
        sub_params : OrderedDict
            Sub-parameters for the model.

        Returns
        -------
        object
            Differential cross-section.
        """
        return self.xs_cal(sub_params, model).dsdo

    def get_diff_xs_vis(self, sub_params, model):
        """
        Get differential cross-section for visualization.

        Parameters
        ----------
        sub_params : OrderedDict
            Sub-parameters for the model.

        Returns
        -------
        object
            Differential cross-section.
        """
        return self.xs_vis(sub_params, model).dsdo

    def get_diff_xs_ratio_to_ruth(self, sub_params, model):
        """
        Get differential cross-section ratio to Rutherford.

        Parameters
        ----------
        sub_params : OrderedDict
            Sub-parameters for the model.

        Returns
        -------
        object
            Differential cross-section ratio.
        """
        xs = self.xs_cal(sub_params, model)
        return xs.dsdo / self.calibration_workspace.rutherford

    def get_diff_xs_ratio_to_ruth_vis(self, sub_params, model):
        """
        Get differential cross-section ratio to Rutherford for visualization.

        Parameters
        ----------
        sub_params : OrderedDict
            Sub-parameters for the model.

        Returns
        -------
        object
            Differential cross-section ratio.
        """
        xs = self.xs_vis(sub_params, model)
        return xs.dsdo / self.visualization_workspace.rutherford


def elm_model(
    sub_params: OrderedDict,
    workspace: jitr.xs.elastic.DifferentialWorkspace,
    reaction: exfor_tools.reaction.Reaction,
):
    """
    ELM model function.

    Parameters
    ----------
    constraint : DifferentialElasticConstraint
        Constraint object.
    sub_params : OrderedDict
        Sub-parameters for the model.
    workspace : jitr.xs.elastic.DifferentialWorkspace
        Workspace for differential calculations.

    Returns
    -------
    object
        Model result.
    """
    if workspace.projectile == (1, 1):
        Ef = reaction.target.Efp
    elif workspace.projectile == (1, 0):
        Ef = reaction.target.Efn
    (
        isoscalar_params,
        isovector_params,
        spin_orbit_params,
        coul_params,
        asym_factor,
    ) = calculate_parameters(
        workspace.projectile,
        workspace.target,
        workspace.Ecm,
        Ef,
        sub_params,
    )

    return workspace.xs(
        interaction_central=central_plus_coulomb,
        interaction_spin_orbit=spin_orbit,
        args_central=(
            workspace.projectile,
            asym_factor,
            isoscalar_params,
            isovector_params,
            coul_params,
        ),
        args_spin_orbit=spin_orbit_params,
    )


def kduq_model(
    sub_params: OrderedDict,
    workspace: jitr.xs.elastic.DifferentialWorkspace,
    reaction: exfor_tools.reaction.Reaction,
):
    """
    KDUQ model function.

    Parameters
    ----------
    constraint : DifferentialElasticConstraint
        Constraint object.
    sub_params : OrderedDict
        Sub-parameters for the model.
    workspace : jitr.xs.elastic.DifferentialWorkspace
        Workspace for differential calculations.
    """
    # TODO
    pass


def wlh_model(
    sub_params: OrderedDict,
    workspace: jitr.xs.elastic.DifferentialWorkspace,
    reaction: exfor_tools.reaction.Reaction,
):
    """
    WLH model function.

    Parameters
    ----------
    constraint : DifferentialElasticConstraint
        Constraint object.
    sub_params : OrderedDict
        Sub-parameters for the model.
    workspace : jitr.xs.elastic.DifferentialWorkspace
        Workspace for differential calculations.
    """
    # TODO
    pass


def set_up_solver(
    reaction: exfor_tools.reaction.Reaction,
    Elab: float,
    x_cal: np.array,
    x_vis: np.array,
    core_solver: jitr.rmatrix.Solver,
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
    x_cal : np.array
        Calibration angles.
    x_vis : np.array
        Visualization angles.
    core_solver : jitr.rmatrix.Solver
        Core solver.
    lmax : int
        Maximum angular momentum.

    Returns
    -------
    tuple
        Calibrator and visualizer workspaces.
    """
    mass_target = reaction.target.m0
    mass_projectile = reaction.projectile.m0

    Zproj = reaction.projectile.Z
    Ztarget = reaction.target.Z

    # get kinematics and parameters for this experiment
    kinematics = jitr.utils.kinematics.semi_relativistic_kinematics(
        mass_target, mass_projectile, Elab, Zproj * Ztarget
    )
    channel_radius_fm = jitr.utils.interaction_range(reaction.projectile.A)
    a = channel_radius_fm * kinematics.k + 2 * np.pi
    Ns = jitr.utils.suggested_basis_size(a)
    N = core_solver.kernel.quadrature.nbasis
    if Ns > N:
        raise ValueError(
            f"Suggested basis size for dimensionless channel radius {a} "
            f"is {Ns}, but core_solver only has {N}"
        )
    sys = jitr.reactions.ProjectileTargetSystem(
        channel_radius=a,
        lmax=lmax,
        mass_target=mass_target,
        mass_projectile=mass_projectile,
        Ztarget=Ztarget,
        Zproj=Zproj,
        coupling=jitr.reactions.system.spin_half_orbit_coupling,
    )

    integral_ws = jitr.xs.elastic.IntegralWorkspace(
        projectile=tuple(reaction.projectile),
        target=tuple(reaction.target),
        sys=sys,
        kinematics=kinematics,
        solver=core_solver,
    )

    calibrator = jitr.xs.elastic.DifferentialWorkspace(
        integral_workspace=integral_ws, angles=x_cal
    )
    visualizer = jitr.xs.elastic.DifferentialWorkspace(
        integral_workspace=integral_ws, angles=x_vis
    )

    return calibrator, visualizer
