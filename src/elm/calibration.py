from collections import OrderedDict
import pickle
from pathlib import Path
import numpy as np
import exfor_tools
import jitr

from .model import calculate_parameters, central_plus_coulomb, spin_orbit


def set_up_solver(
    projectile: tuple,
    target: tuple,
    Elab: float,
    angles_cal: np.array,
    angles_vis: np.array,
    core_solver: jitr.rmatrix.Solver,
    channel_radius: float,
    lmax: int,
):

    sys = jitr.reactions.ProjectileTargetSystem(
        channel_radius=channel_radius,
        lmax=lmax,
        mass_target=jitr.utils.kinematics.mass(*target),
        mass_projectile=jitr.utils.kinematics.mass(*projectile),
        Ztarget=target[1],
        Zproj=projectile[1],
        coupling=jitr.reactions.system.spin_half_orbit_coupling,
    )

    # get kinematics and parameters for this experiment
    kinematics = jitr.utils.kinematics.semi_relativistic_kinematics(
        sys.mass_target, sys.mass_projectile, Elab, sys.Zproj * sys.Ztarget
    )

    integral_ws = jitr.xs.elastic.IntegralWorkspace(
        projectile=projectile,
        target=target,
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


class DifferentialXS:
    def __init__(
        self,
        projectile: tuple,
        target: tuple,
        Elab: float,
        exp: exfor_tools.ExforDifferentialDataSet,
        angles_vis: np.array,
        core_solver: jitr.rmatrix.Solver,
        channel_radius: float,
        lmax: int = 20,
        absolute_xs=True,
    ):
        self.exp = exp
        self.N = self.exp.data.shape[1]
        self.angles_vis = angles_vis
        self.angles_cal = exp.data[0, :] * np.pi / 180

        calibrator, visualizer = set_up_solver(
            projectile,
            target,
            Elab,
            self.angles_cal,
            self.angles_vis,
            core_solver,
            channel_radius,
            lmax,
        )
        self.calibration_model = calibrator
        self.visualization_model = visualizer
        self.target = self.calibration_model.target
        self.projectile = self.calibration_model.projectile
        self.Elab = self.exp.Elab

        if absolute_xs and self.projectile[1] > 0:
            self.exp.data[2, :] /= self.calibration_model.rutherford
            self.exp.data[3, :] /= self.calibration_model.rutherford

        if projectile == (1, 1):
            self.Ef = jitr.utils.kinematics.proton_fermi_energy(*target)
            self.get_y = self.get_diff_xs_ratio_to_ruth
        elif projectile == (1, 0):
            self.Ef = jitr.utils.kinematics.neutron_fermi_energy(*target)
            self.get_y = self.get_diff_xs
        else:
            raise NotImplementedError("Only neutron and proton projectiles are valid")

    def params(self, sub_params: OrderedDict):
        return calculate_parameters(
            self.calibration_model.projectile,
            self.calibration_model.target,
            self.calibration_model.Ecm,
            self.Ef,
            sub_params,
        )

    def xs(self, sub_params: OrderedDict, model: jitr.xs.elastic.DifferentialWorkspace):
        (
            isoscalar_params,
            isovector_params,
            spin_orbit_params,
            coul_params,
            asym_factor,
        ) = self.params(sub_params)
        return model.xs(
            interaction_central=central_plus_coulomb,
            interaction_spin_orbit=spin_orbit,
            args_central=(
                self.calibration_model.projectile,
                asym_factor,
                isoscalar_params,
                isovector_params,
                coul_params,
            ),
            args_spin_orbit=spin_orbit_params,
        )

    def xs_model(self, sub_params: OrderedDict):
        return self.xs(sub_params, self.calibration_model)

    def xs_vis(self, sub_params: OrderedDict):
        return self.xs(sub_params, self.visualization_model)

    def get_diff_xs(self, xs: jitr.xs.elastic.ElasticXS):
        return xs.dsdo

    def get_diff_xs_ratio_to_ruth(self, xs: jitr.xs.elastic.ElasticXS):
        return xs.dsdo / xs.rutherford

    def y_model(self, sub_params: OrderedDict):
        xs = self.xs_model(sub_params)
        return self.get_y(xs)

    def y_vis(self, sub_params: OrderedDict):
        xs = self.xs_vis(sub_params)
        return self.get_y(xs)

    def residual(self, sub_params: OrderedDict):
        y_model = self.y_model(sub_params)
        y_exp = self.exp.data[2, :]
        return y_model - y_exp

    def sigma_y(
        self,
    ):
        return self.exp.data[3, :]
