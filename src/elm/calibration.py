from collections import OrderedDict

import numpy as np

import exfor_tools
import jitr

from .model import calculate_parameters, central_plus_coulomb, spin_orbit


class Constraint:
    """Represents an experimental constraint y, with a covariance matrix,
    and some model for y, f, that takes in some params"""

    def __init__(self, n_params: int, y: np.ndarray, covariance: np.ndarray):
        self.n_params = n_params
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
        pass

    def residual(self, params):
        return self.y - self.f(params)

    def chi2(self, params):
        delta = self.residual(params)
        return delta.T @ self.cov_inv @ delta


class ChExIASConstraint(Constraint):
    pass  # TODO


class DifferentialElasticConstraint(Constraint):
    def __init__(
        self,
        n_params: int,
        model,
        reaction: exfor_tools.Reaction,
        quantity: str,
        measurement: exfor_tools.AngularDistributionSysStatErr,
        angles_vis: np.array,
        core_solver: jitr.rmatrix.Solver,
        channel_radius: float,
        lmax: int = 20,
        absolute_xs=True,
    ):
        self.reaction = reaction
        self.quantity = quantity
        self.model = model
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

        self.x_vis = angles_vis
        self.x_cal = self.measurement.x * np.pi / 180
        if self.measurement.x_units != "degrees":
            raise ValueError(f"invalid x units {self.measurement.x_units}")

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
        super().__init__(self.n_params, self.measurement.y, covariance)

        calibrator, visualizer = set_up_solver(
            self.reaction,
            self.measurement.Einc,
            self.angles_cal,
            self.angles_vis,
            core_solver,
            channel_radius,
            lmax,
        )
        self.calibration_workspace = calibrator
        self.visualization_workspace = visualizer

        self.Elab = self.calibration_workspace.Elab
        self.mu = self.calibration_workspace.mu
        self.Ecm = self.calibration_workspace.Ecm
        self.k = self.calibration_workspace.k
        self.eta = self.calibration_workspace.eta

        if absolute_xs and self.projectile[1] > 0:
            self.exp.data[2, :] /= self.calibration_workspace.rutherford
            self.exp.data[3, :] /= self.calibration_workspace.rutherford

        if self.quantity == "dXS/dA":
            self.f = self.get_diff_xs
            self.f_vis = self.get_diff_xs_vis
        elif self.quantity == "dXS/dRuth":
            if self.reaction.projectile[1] == 0:
                raise ValueError("Can't do dXS/dRuth for uncharged projectile")
            self.f = self.get_diff_xs_ratio_to_ruth
            self.f_vis = self.get_diff_xs_ratio_to_ruth_vis
        elif self.quantity == "Ay":
            self.f = self.get_Ay
            self.f_vis = self.get_Ay_vis
        else:
            raise NotImplementedError(f"{self.quantity} not supported")

        if self.reaction.projectile == (1, 1):
            self.Ef = jitr.utils.kinematics.proton_fermi_energy(*self.reaction.target)
        elif self.reaction.projectile == (1, 0):
            self.Ef = jitr.utils.kinematics.neutron_fermi_energy(*self.reaction.target)
        else:
            raise NotImplementedError("Only neutron and proton projectiles are valid")

    def xs_cal(self, sub_params):
        return self.model(self, sub_params, self.calibration_workspace)

    def xs_vis(self, sub_params):
        return self.model(self, sub_params, self.visualization_workspace)

    def get_Ay(self, sub_params):
        return self.xs_cal(sub_params).Ay

    def get_Ay_vis(self, sub_params):
        return self.xs_vis(sub_params).Ay

    def get_diff_xs(self, sub_params):
        return self.xs_cal(sub_params).dsdo

    def get_diff_xs_vis(self, sub_params):
        return self.xs_vis(sub_params).dsdo

    def get_diff_xs_ratio_to_ruth(self, sub_params):
        xs = self.xs_cal(sub_params)
        return xs.dsdo / self.calibration_workspace.rutherford

    def get_diff_xs_ratio_to_ruth_vis(self, sub_params):
        xs = self.xs_vis(sub_params)
        return xs.dsdo / self.visualization_workspace.rutherford


def elm_model(
    constraint: DifferentialElasticConstraint,
    sub_params: OrderedDict,
    workspace: jitr.xs.elastic.DifferentialWorkspace,
):

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
        constraint.Ef,
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
    constraint: DifferentialElasticConstraint,
    sub_params: OrderedDict,
    workspace: jitr.xs.elastic.DifferentialWorkspace,
):
    # TODO
    pass


def wlh_model(
    constraint: DifferentialElasticConstraint,
    sub_params: OrderedDict,
    workspace: jitr.xs.elastic.DifferentialWorkspace,
):
    # TODO
    pass


def set_up_solver(
    reaction: exfor_tools.Reaction,
    Elab: float,
    angles_cal: np.array,
    angles_vis: np.array,
    core_solver: jitr.rmatrix.Solver,
    lmax: int,
):

    mass_target = jitr.utils.kinematics.mass(*reaction.target)
    mass_projectile = jitr.utils.kinematics.mass(*reaction.projectile)

    Zproj = reaction.projectile[1]
    Ztarget = reaction.target[1]

    # get kinematics and parameters for this experiment
    kinematics = jitr.utils.kinematics.semi_relativistic_kinematics(
        mass_target, mass_projectile, Elab, Zproj * Ztarget
    )
    channel_radius_fm = jitr.utils.interaction_range(reaction.projectile[0])
    a = channel_radius_fm * kinematics.k + 2 * np.pi
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
        projectile=reaction.projectile,
        target=reaction.target,
        sys=sys,
        kinematics=kinematics,
        solver=core_solver,
    )

    calibrator = jitr.xs.elastic.DifferentialWorkspace(
        integral_workspace=integral_ws, angles=angles_cal
    )
    visualizer = jitr.xs.elastic.DifferentialWorkspace(
        integral_workspace=integral_ws, angles=angles_vis
    )

    return calibrator, visualizer


