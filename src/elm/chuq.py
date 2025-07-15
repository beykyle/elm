"""The CHUQ potential is a global phenomenological nucleon-nucleus
optical potential

See [Pruitt, et al., 2023]
(https://journals.aps.org/prc/pdf/10.1103/PhysRevC.107.014602), or
the original CH89 paper [Varner, et al., 1991]
(https://www.sciencedirect.com/science/article/pii/037015739190039O?via%3Dihub)
for details. Equation references are with respect to the former paper.
"""

from collections import OrderedDict
from pathlib import Path
import json

import numpy as np

from rxmc.params import Parameter

from jitr.utils.constants import ALPHA, HBARC
from jitr.optical_potentials.potential_forms import (
    thomas_safe,
    woods_saxon_prime_safe,
    woods_saxon_safe,
    coulomb_charged_sphere,
)

params = [
    Parameter("V0", np.float64, "MeV", "V_0"),
    Parameter("Ve", np.float64, "no-dim", "V_e"),
    Parameter("Vt", np.float64, "MeV", "V_t"),
    Parameter("r0", np.float64, "fm", "r_0"),
    Parameter("r0_0", np.float64, "fm", "r_0^{(0)}"),
    Parameter("a0", np.float64, "fm", "a_0"),
    Parameter("Wv0", np.float64, "MeV", "W_{v0}"),
    Parameter("Wve0", np.float64, "MeV", "W_{ve0}"),
    Parameter("Wvew", np.float64, "MeV", "W_{vew}"),
    Parameter("rw", np.float64, "fm", "r_w"),
    Parameter("rw_0", np.float64, "fm", "r_w^{(0)}"),
    Parameter("aw", np.float64, "fm", "a_w"),
    Parameter("Ws0", np.float64, "MeV", "W_{s0}"),
    Parameter("Wst", np.float64, "MeV", "W_{st}"),
    Parameter("Wse0", np.float64, "MeV", "W_{se0}"),
    Parameter("Wsew", np.float64, "MeV", "W_{sew}"),
    Parameter("Vso", np.float64, "MeV", "V_{so}"),
    Parameter("rso", np.float64, "fm", "r_{so}"),
    Parameter("rso_0", np.float64, "fm", "r_{so}^{(0)}"),
    Parameter("aso", np.float64, "fm", "a_{so}"),
    Parameter("rc", np.float64, "fm", "r_c"),
    Parameter("rc_0", np.float64, "fm", "r_c^{(0)}"),
]
NUM_PARAMS = len(params)


def central_form(r, V, W, Wd, R, a, Rd, ad):
    r"""form of central part (volume and surface)"""
    volume = V * woods_saxon_safe(r, R, a)
    imag_volume = 1j * W * woods_saxon_safe(r, Rd, ad)
    surface = -(4j * ad * Wd) * woods_saxon_prime_safe(r, Rd, ad)
    return -volume - imag_volume - surface


def spin_orbit_form(r, Vso, Rso, aso):
    r"""form of spin-orbit term"""
    return 2 * Vso * thomas_safe(r, Rso, aso)


def central(r, asym_factor, isoscalar_params, isovector_params):
    V0 = central_form(r, *isoscalar_params)
    V1 = central_form(r, *isovector_params)
    return V0 + asym_factor * V1


def central_plus_coulomb(
    r,
    asym_factor,
    coulomb_params,
    central_isoscalar_params,
    central_isovector_params,
):
    r"""sum of coulomb, central isoscalar and central isovector terms"""
    coulomb = coulomb_charged_sphere(r, *coulomb_params)
    centr = central(r, asym_factor, central_isoscalar_params, central_isovector_params)
    return centr + coulomb


def spin_orbit(r, asym_factor, isoscalar_params, isovector_params):
    Vso0 = spin_orbit_form(r, *isoscalar_params)
    Vso1 = spin_orbit_form(r, *isovector_params)
    return Vso0 + asym_factor * Vso1


def calculate_params(
    projectile: tuple,
    target: tuple,
    Elab: float,
    V0: float = 52.9,
    Ve: float = -0.299,
    Vt: float = 13.1,
    r0: float = 1.25,
    r0_0: float = -0.225,
    a0: float = 0.69,
    Wv0: float = 7.8,
    Wve0: float = 35.0,
    Wvew: float = 16.0,
    rw: float = 1.33,
    rw_0: float = -0.42,
    aw: float = 0.69,
    Ws0: float = 10.0,
    Wst: float = 18.0,
    Wse0: float = 36.0,
    Wsew: float = 37,
    Vso: float = 5.9,
    rso: float = 1.34,
    rso_0: float = -1.2,
    aso: float = 0.63,
    rc: float = 1.24,
    rc_0: float = 0.12,
):
    """
    Calculate the parameters for the optical model potential.

    Parameters:
    ----------
    projectile : tuple
        A tuple containing the mass number and charge of the projectile.
    target : tuple
        A tuple containing the mass number and charge of the target.
    Elab : float
        The laboratory energy of the projectile in MeV.
    V0, Ve, ..., rc_0: float
        Parameters for the Chapel-Hill optical model potential.
        See Table V and the Appendix
        of [Pruitt, et al., 2023]
        (https://journals.aps.org/prc/pdf/10.1103/PhysRevC.107.014602)
        for details.

        See also Table 3. of the original CH89 paper [Varner, et al., 1991]
        (https://www.sciencedirect.com/science/article/pii/037015739190039O?via%3Dihub).

        Note: there is a typo in Eq. A4 of the CHUQ paper, there should be a
        plus/minus sign in front of Wst, not a plus. See Table 3 of the
        original CH89 paper for the correct sign.

    Returns:
        central isoscalar_params (tuple): (V0, W0, Wd0, R0, a0, Rd, ad)
        central isovector_params (tuple): (V1, W1, Wd1, R1, a1, Rd1, ad1)
        spin orbit isocalar params (tuple): (Vso0, rso0, aso0)
        spin orbit isovector params (tuple): (Vso1, rso1, aso1)
        Coulomb_params (tuple): (Zz, Rc)
        asym_factor =  +(-) (N-Z)/(N+Z), for neutrons(protons)
    """
    A, Z = target
    Ap, Zp = projectile
    is_proton = Zp == 1
    N = A - Z
    assert Ap == 1 and Zp in (0, 1)

    # Asymmetry factor
    alpha = (N - Z) / A
    sign = 1 if is_proton else -1
    asym_factor = sign * alpha

    # Radii (Eq. A6)
    R0 = r0 * A ** (1 / 3) + r0_0
    Rw = rw * A ** (1 / 3) + rw_0
    Rso = rso * A ** (1 / 3) + rso_0
    RC = rc * A ** (1 / 3) + rc_0

    # Coulomb correction
    Ec = coulomb_correction(A, Z, RC) if is_proton else 0.0
    delta_E = Elab - Ec

    # Real central depths (Eq. A4)
    V0 = V0 + Ve * delta_E
    Vt = Vt  # no energy dependence for isovector real central term

    # Imaginary depths (Eq. A4)
    Wv = Wv0 / (1 + np.exp((Wve0 - delta_E) / Wvew))
    Ws = Ws0 / (1 + np.exp((delta_E - Wse0) / Wsew))
    Wst = Wst / (1 + np.exp((delta_E - Wse0) / Wsew))

    central_isoscalar_params = (V0, Wv, Ws, R0, a0, Rw, aw)
    spin_orbit_isoscalar_params = (Vso, Rso, aso)
    central_isovector_params = (Vt, 0, Wst, R0, a0, Rw, aw)
    spin_orbit_isovector_params = (0, Rso, aso)
    coulomb_params = (Z, RC)

    return (
        central_isoscalar_params,
        central_isovector_params,
        spin_orbit_isoscalar_params,
        spin_orbit_isovector_params,
        coulomb_params,
        asym_factor,
    )


def coulomb_correction(A, Z, RC):
    r"""
    Coulomb correction for proton energy
    """
    return 6.0 * Z * ALPHA * HBARC / (5 * RC)


class Global:
    r"""Global optical potential in CHUQ form."""

    def __init__(self, projectile: tuple, param_fpath: Path = None):
        r"""
        Parameters:
            projectile : neutron or proton?
            param_fpath : path to json file encoding parameter values.
                Defaults to data/WLH_mean.json
        """
        if param_fpath is None:
            param_fpath = Path(__file__).parent.resolve() / Path(
                "./../../data/CH89_default.json"
            )

        if projectile not in [(1, 0), (1, 1)]:
            raise RuntimeError(
                "chuq.Global is defined only for neutron and proton projectiles"
            )

        self.params = OrderedDict()
        self.projectile = projectile

        self.param_fpath = param_fpath
        with open(self.param_fpath) as f:
            data = json.load(f)

            if "CH89RealCentral" in data:
                self.params["V0"] = data["CH89RealCentral"]["V_0"]
                self.params["Ve"] = data["CH89RealCentral"]["V_e"]
                self.params["Vt"] = data["CH89RealCentral"]["V_t"]
                self.params["r0"] = data["CH89RealCentral"]["r_o"]
                self.params["r0_0"] = data["CH89RealCentral"]["r_o_0"]
                self.params["a0"] = data["CH89RealCentral"]["a_0"]

                self.params["Wv0"] = data["CH89ImagCentral"]["W_v0"]
                self.params["Wve0"] = data["CH89ImagCentral"]["W_ve0"]
                self.params["Wvew"] = data["CH89ImagCentral"]["W_vew"]
                self.params["rw"] = data["CH89ImagCentral"]["r_w"]
                self.params["rw_0"] = data["CH89ImagCentral"]["r_w0"]
                self.params["aw"] = data["CH89ImagCentral"]["a_w"]
                self.params["Ws0"] = data["CH89ImagCentral"]["W_s0"]
                self.params["Wst"] = data["CH89ImagCentral"]["W_st"]
                self.params["Wse0"] = data["CH89ImagCentral"]["W_se0"]
                self.params["Wsew"] = data["CH89ImagCentral"]["W_sew"]

                self.params["Vso"] = data["CH89SpinOrbit"]["V_so"]
                self.params["rso"] = data["CH89SpinOrbit"]["r_so"]
                self.params["rso_0"] = data["CH89SpinOrbit"]["r_so_0"]
                self.params["aso"] = data["CH89SpinOrbit"]["a_so"]

                self.params["rc"] = data["CH89Coulomb"]["r_c"]
                self.params["rc_0"] = data["CH89Coulomb"]["r_c_0"]

            elif "CH89RealCentral_V_0" in data:

                self.params["V0"] = data["CH89RealCentral_V_0"]
                self.params["Ve"] = data["CH89RealCentral_V_e"]
                self.params["Vt"] = data["CH89RealCentral_V_t"]
                self.params["r0"] = data["CH89RealCentral_r_o"]
                self.params["r0_0"] = data["CH89RealCentral_r_o_0"]
                self.params["a0"] = data["CH89RealCentral_a_0"]
                self.params["Wv0"] = data["CH89ImagCentral_W_v0"]
                self.params["Wve0"] = data["CH89ImagCentral_W_ve0"]
                self.params["Wvew"] = data["CH89ImagCentral_W_vew"]
                self.params["rw"] = data["CH89ImagCentral_r_w"]
                self.params["rw_0"] = data["CH89ImagCentral_r_w0"]
                self.params["aw"] = data["CH89ImagCentral_a_w"]
                self.params["Ws0"] = data["CH89ImagCentral_W_s0"]
                self.params["Wst"] = data["CH89ImagCentral_W_st"]
                self.params["Wse0"] = data["CH89ImagCentral_W_se0"]
                self.params["Wsew"] = data["CH89ImagCentral_W_sew"]
                self.params["Vso"] = data["CH89SpinOrbit_V_so"]
                self.params["rso"] = data["CH89SpinOrbit_r_so"]
                self.params["rso_0"] = data["CH89SpinOrbit_r_so_0"]
                self.params["aso"] = data["CH89SpinOrbit_a_so"]
                self.params["rc"] = data["CH89Coulomb_r_c"]
                self.params["rc_0"] = data["CH89Coulomb_r_c_0"]

            else:
                raise ValueError("Unrecognized parameter file format for WLH!")

    def get_params(self, A, Z, Elab):
        # fermi energy
        return calculate_params(
            self.projectile, (A, Z), Elab, *list(self.params.values())
        )


def chuq_elastic(ws, *x):
    r"""
    Callable for
    `rxmc.ElasticDifferentialXSModel.calculate_interaction_from_params`

    Given a workspace and a set of CHUQ parameters, returns the args for
    `elm.model_form.central_plus_coulomb` and `elm.model_form.spin_orbit`

    Parameters
    ----------
    ws : Workspace
        The workspace containing the reaction and kinematics.
    x : tuple
        The parameters for CHUQ
    """
    rxn = ws.reaction
    kinematics = ws.kinematics
    (
        central_isoscalar_params,
        central_isovector_params,
        spin_orbit_isoscalar_params,
        spin_orbit_isovector_params,
        coul_params,
        asym_factor,
    ) = calculate_params(tuple(rxn.projectile), tuple(rxn.target), kinematics.Elab, *x)
    args_central = (
        asym_factor,
        coul_params,
        central_isoscalar_params,
        central_isovector_params,
    )
    args_spin_orbit = (
        asym_factor,
        spin_orbit_isoscalar_params,
        spin_orbit_isovector_params,
    )
    return args_central, args_spin_orbit
