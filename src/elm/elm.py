from collections import OrderedDict
import numpy as np

from rxmc.params import Parameter

from jitr.optical_potentials.potential_forms import (
    thomas_safe,
    woods_saxon_prime_safe,
    woods_saxon_safe,
    coulomb_charged_sphere,
)
from jitr.utils.constants import ALPHA, HBARC, WAVENUMBER_PION
from jitr import xs


params = [
    Parameter("V0", np.float64, r"MeV", r"V_0"),
    Parameter("W0", np.float64, r"MeV", r"W_0"),
    Parameter("Wd0", np.float64, r"MeV", r"W_{D0}"),
    Parameter("V1", np.float64, r"MeV", r"V_1"),
    Parameter("W1", np.float64, r"MeV", r"W_1"),
    Parameter("Wd1", np.float64, r"MeV", r"W_{D1}"),
    #   Parameter("eta", np.float64, r"dimensionless", r"\eta"),
    Parameter("alpha", np.float64, r"dimensionless", r"\alpha"),
    #   Parameter("beta", np.float64, r"MeV$^{-2}$", r"\beta"),
    Parameter("gamma_w", np.float64, r"MeV", r"\gamma_W"),
    Parameter("gamma_d", np.float64, r"MeV", r"\gamma_D"),
    Parameter("r0", np.float64, r"fm", r"r_0"),
    Parameter("r1", np.float64, r"fm", r"r_1"),
    Parameter("r0A", np.float64, r"fm", r"r_{0A}"),
    Parameter("r1A", np.float64, r"fm", r"r_{1A}"),
    Parameter("a0", np.float64, r"fm", r"a_0"),
    Parameter("a1", np.float64, r"fm", r"a_1"),
]
params_dtype = [(p.name, p.dtype) for p in params]
NUM_PARAMS = len(params)


def central_form(r, V, W, Wd, R, a, Rd, ad):
    r"""form of central part (volume and surface) as a function of radial distance"""
    volume = -(V + 1j * W) * woods_saxon_safe(r, R, a)
    surface = (4j * ad * Wd) * woods_saxon_prime_safe(r, Rd, ad)
    return volume + surface


def spin_orbit_form(r, Vso, Wso, R, a):
    r"""form of spin-orbit term"""
    # extra factor of 2 comes from use of l dot s rather than l dot sigma
    return 2 * (Vso + 1j * Wso) * (1 / WAVENUMBER_PION) ** 2 * thomas_safe(r, R, a)


def spin_orbit(
    r, asym_factor, spin_orbit_isoscalar_params, spin_orbit_isovector_params
):
    """sum of spin-orbit isoscalar and isovector terms"""
    Vso0 = spin_orbit_form(r, *spin_orbit_isoscalar_params)
    Vso1 = spin_orbit_form(r, *spin_orbit_isovector_params)
    return Vso0 + asym_factor * Vso1


def central_plus_coulomb(
    r,
    asym_factor,
    central_isoscalar_params,
    central_isovector_params,
    coulomb_params,
):
    r"""sum of coulomb, central isoscalar and central isovector terms"""
    coulomb = coulomb_charged_sphere(r, *coulomb_params)
    centr = central(r, asym_factor, central_isoscalar_params, central_isovector_params)
    return centr + coulomb


def central(r, asym_factor, central_isoscalar_params, central_isovector_params):
    r"""sum of central isoscalar and central isovector terms"""
    V0 = central_form(r, *central_isoscalar_params)
    V1 = central_form(r, *central_isovector_params)
    centr = V0 + asym_factor * V1
    return centr


def coulomb_correction(A, Z, RC):
    r"""
    Coulomb correction for proton energy
    """
    return 6.0 * Z * ALPHA * HBARC / (5 * RC)


def calculate_parameters(
    projectile: tuple, target: tuple, E: float, Ef: float, params: OrderedDict
):
    r"""Calculate the parameters in the ELM for a given target isotope
    and energy, given a subparameter sample
    Parameters:
        projectile (tuple): projectile A,Z - must be neutron or proton ((1,0) or (1,1))
        target (tuple): target A,Z
        E (float): center-of-mass frame energy
        Ef  (float): Fermi energy for A,Z nucleus
        params (OrderedDict) : subparameter sample
    Returns:
        central isoscalar_params (tuple): (V0, W0, Wd0, R0, a0, Rd, ad)
        central isovector_params (tuple): (V1, W1, Wd1, R1, a1, Rd1, ad1)
        spin orbit isocalar params (tuple): (Vso0, rso0, aso0)
        spin orbit isovector params (tuple): (Vso1, rso1, aso1)
        Coulomb_params (tuple): (Zz, Rc)
        asym_factor =  +(-) (N-Z)/(N+Z), for neutrons(protons)
    """
    # asymmetry for central_isovector dependence
    A, Z = target
    Ap, Zp = projectile
    assert Ap == 1 and (Zp == 1 or Zp == 0)
    asym_factor = (A - 2 * Z) / (A)
    asym_factor *= (-1) ** (Zp)

    # geometries
    R0 = params["r0"] + params["r0A"] * A ** (1.0 / 3.0)
    R1 = params["r1"] + params["r1A"] * A ** (1.0 / 3.0)
    a0 = params["a0"]
    a1 = params["a1"]

    # Coulomb radius equal to central isoscalar radius
    RC = R0

    # energy
    dE = E - Ef
    if projectile == (1, 1):
        dE -= coulomb_correction(A, Z, RC)

    # energy dependence of depths
    erg_v = 1 + params["alpha"] / params["V0"] * dE
    erg_w = dE**2 / (dE**2 + params["gamma_w"] ** 2)
    erg_wd = dE**2 / (dE**2 + params["gamma_d"] ** 2)

    # central isoscalar depths
    V0 = params["V0"] * erg_v
    W0 = params["W0"] * erg_w
    Wd0 = params["Wd0"] * erg_wd

    # central isovector depths
    V1 = params["V1"] * erg_v
    W1 = params["W1"] * erg_w
    Wd1 = params["Wd1"] * erg_wd

    # spin orbit depths are just a fixed ratio (eta) of central depths
    # for now, keep this fixed
    # eta = 0.44  # params["eta"]

    # alternative option just fixing all so depths to be A,E independent:
    Vso0 = 5.58
    Wso0 = 0
    Vso1 = 0
    Wso1 = 0

    # spin orbit isovector depths
    # Vso0 = V0 * eta
    # fix at KDUQ value but convert from form using
    # (hbar/mpi c)^2 * l.sigma to 1/r0^2 * (l.s)
    # Wso0 = W0 * eta

    # spin orbit isovector depths
    # Vso1 = V1 * eta
    # Wso1 = W1 * eta

    return (
        (V0, W0, Wd0, R0, a0, R0, a0),
        (V1, W1, Wd1, R1, a1, R1, a1),
        (Vso0, Wso0, R0, a0),
        (Vso1, Wso1, R1, a1),
        (Z, RC),
        asym_factor,
    )


def calculate_chex_ias_differential_xs(
    workspace: xs.quasielastic_pn.Workspace,
    params: OrderedDict,
):
    rxn = workspace.reaction
    assert rxn.projectile == (1, 1)
    assert rxn.product == (1, 0)
    (
        p_central_isoscalar_params,
        p_central_isovector_params,
        p_spin_orbit_isoscalar_params,
        p_spin_orbit_isovector_params,
        p_coul_params,
        p_asym_factor,
    ) = calculate_parameters(
        projectile=tuple(rxn.projectile),
        target=tuple(rxn.target),
        E=workspace.kinematics_entrance.Ecm,
        Ef=rxn.target.Efp,
        params=params,
    )
    (
        n_central_isoscalar_params,
        n_central_isovector_params,
        n_spin_orbit_isoscalar_params,
        n_spin_orbit_isovector_params,
        _,
        n_asym_factor,
    ) = calculate_parameters(
        projectile=tuple(rxn.residual),
        target=tuple(rxn.product),
        E=workspace.kinematics_exit.Ecm,
        Ef=rxn.residual.Efn,
        params=params,
    )

    return workspace.xs(
        U_p_coulomb=coulomb_charged_sphere,
        U_p_scalar=central,
        U_p_spin_orbit=spin_orbit,
        U_n_central=central,
        U_n_spin_orbit=spin_orbit,
        args_p_coulomb=p_coul_params,
        args_p_scalar=(
            p_asym_factor,
            p_central_isoscalar_params,
            p_central_isovector_params,
        ),
        args_p_spin_orbit=(
            p_asym_factor,
            p_spin_orbit_isoscalar_params,
            p_spin_orbit_isovector_params,
        ),
        args_n_scalar=(
            n_asym_factor,
            n_central_isoscalar_params,
            n_central_isovector_params,
        ),
        args_n_spin_orbit=(
            n_asym_factor,
            n_spin_orbit_isoscalar_params,
            n_spin_orbit_isovector_params,
        ),
    )


def calculate_elastic_differential_xs(
    workspace: xs.elastic.DifferentialWorkspace,
    params: OrderedDict,
):
    rxn = workspace.reaction
    kinematics = workspace.kinematics
    assert rxn.projectile.A == 1
    (
        central_isoscalar_params,
        central_isovector_params,
        spin_orbit_isoscalar_params,
        spin_orbit_isovector_params,
        coul_params,
        asym_factor,
    ) = calculate_parameters(
        projectile=tuple(rxn.projectile),
        target=tuple(rxn.target),
        E=kinematics.Ecm,
        Ef=rxn.Ef,
        params=params,
    )
    return workspace.xs(
        interaction_central=central_plus_coulomb,
        interaction_spin_orbit=spin_orbit,
        args_central=(
            asym_factor,
            central_isoscalar_params,
            central_isovector_params,
            coul_params,
        ),
        args_spin_orbit=(
            asym_factor,
            spin_orbit_isoscalar_params,
            spin_orbit_isovector_params,
        ),
    )
