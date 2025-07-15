import numpy as np
import jitr
from jitr.optical_potentials.potential_forms import (
    thomas_safe,
    woods_saxon_prime_safe,
    woods_saxon_safe,
    coulomb_charged_sphere,
)
from jitr.utils.constants import WAVENUMBER_PION, ALPHA, HBARC


# TODO
# - add Coulomb correction
# - use R_C = R_p adjusted from R_0 and R_1
# - use Fermi energy


class Parameter:
    def __init__(self, name, dtype, fancy_label=None, bounds=None, note=None):
        self.name = name
        self.dtype = dtype
        self.fancy_label = fancy_label if fancy_label is not None else name
        self.bounds = bounds
        self.note = note


elm_params = [
    Parameter("V0", np.float64, r"$V_0$ [MeV]"),
    Parameter("W0", np.float64, r"$W_0$ [MeV]"),
    Parameter("Wd0", np.float64, r"$W_{D0}$ [MeV]"),
    Parameter("V1", np.float64, r"$V_1$ [MeV]"),
    Parameter("W1", np.float64, r"$W_1$ [MeV]"),
    Parameter("Wd1", np.float64, r"$W_{D1}$ [MeV]"),
    Parameter("Vso", np.float64, r"$V_{so}$ [MeV]"),
    Parameter("alpha", np.float64, r"$\alpha$"),
    Parameter("beta", np.float64, r"$\beta$"),
    Parameter("gamma_w", np.float64, r"$\gamma_W$ [MeV]"),
    Parameter("gamma_d", np.float64, r"$\gamma_D$ [MeV]"),
    Parameter("r0", np.float64, r"$r_0$ [fm]", bounds=[0, np.inf]),
    Parameter("r1", np.float64, r"$r_1$ [fm]", bounds=[0, np.inf]),
    Parameter("a0", np.float64, r"$a_0$ [fm]", bounds=[0, np.inf]),
    Parameter("a1", np.float64, r"$a_1$ [fm]", bounds=[0, np.inf]),
]
elm_params_dtype = [(p.name, p.dtype) for p in elm_params]
elm_fancy_labels = dict([(p.name, p.fancy_label) for p in elm_params])
N_PARAMS = len(elm_params)


def elm_isoscalar(r, V0, W0, Wd0, R0, a0):
    r"""isoscalar part of the EL model (without spin-orbit) as a function of radial distance r"""
    return -(V0 + 1j * W0) * woods_saxon_safe(r, R0, a0) + (
        4j * a0 * Wd0
    ) * woods_saxon_prime_safe(r, R0, a0)


def elm_isovector(r, V1, W1, Wd1, R1, a1):
    r"""isovector part of the EL model (without spin-orbit) as a function of radial distance r"""
    return -(V1 + 1j * W1) * woods_saxon_safe(r, R1, a1) + (
        4j * a1 * Wd1
    ) * woods_saxon_prime_safe(r, R1, a1)


def elm_so(r, Vso, R0, a0):
    r"""ELM spin-orbit term"""
    return Vso / WAVENUMBER_PION**2 * thomas_safe(r, R0, a0)


def el_model_params(projectile, A, Z, E, sp):
    r"""Calculate the parameters in the ELM for a given target isotope
        and energy
    l   Parameters:
            A (int): target mass
            Z (int): target charge
            E (int): center-of-mass frame energy
            sp (tuple) : sub parameters in the form of a tuple in specified order
        Returns:
            isoscalar_params (tuple)
            isovector_params (tuple)
            spin_orbit_params (tuple)
            Coulomb_params (tuple)
            delta =  (N-Z)/(N+Z)
    """
    V0, W0, Wd0, V1, W1, Wd1, Vso, alpha, beta, gamma_w, gamma_d, r0, r1, a0, a1 = sp
    delta = (A - 2 * Z) / (A)
    # dE = E - Ef
    if projectile == (1, 1):
        dE = E  # - elm_coulomb_correction(A,Z,E,sp)
    elif projectile == (1, 0):
        dE = E
    erg_v = 1.0 + alpha * dE + beta * dE**2
    erg_w = dE**2 / (dE**2 + gamma_w**2)
    erg_wd = dE**2 / (dE**2 + gamma_d**2) * np.exp(-dE / gamma_d)
    V0 = V0 * erg_v
    W0 = W0 * erg_w
    Wd0 = Wd0 * erg_wd
    Vso = 5.58
    V1 = V1 * erg_v
    W1 = W1 * erg_w
    Wd1 = Wd1 * erg_wd
    R0 = r0 * A ** (1.0 / 3.0)
    R1 = r1 * A ** (1.0 / 3.0)
    a0 = a0
    a1 = a1
    return (
        (V0, W0, Wd0, R0, a0),
        (V1, W1, Wd1, R1, a1),
        (Vso, R0, a0),
        (Z, R0),
        delta,
    )


def elm_spin_scalar_plus_coulomb(
    r, projectile, alpha, isoscalar_params, isovector_params, coulomb_params
):
    if projectile == (1, 1):
        factor = 1
        coul = coulomb_charged_sphere(r, *coulomb_params)
    elif projectile == (1, 0):
        factor = -1
        coul = 0

    nucl = elm_isoscalar(r, *isoscalar_params) + factor * alpha * elm_isovector(
        r, *isovector_params
    )
    return nucl + coul


def elm_spin_scalar(
    r,
    projectile,
    alpha,
    isoscalar_params,
    isovector_params,
):
    if projectile == (1, 1):
        factor = 1
    elif projectile == (1, 0):
        factor = -1

    nucl = elm_isoscalar(r, *isoscalar_params) + factor * alpha * elm_isovector(
        r, *isovector_params
    )
    return nucl


def elm_coulomb_correction(A, Z, E, sp):
    r"""
    Coulomb correction for proton energy
    """
    V0, W0, Wd0, V1, W1, Wd1, Vso, alpha, beta, gamma_w, gamma_d, r0, r1, a0, a1 = sp
    R0 = r0 * A ** (1.0 / 3.0)
    return 6.0 * Z * ALPHA * HBARC / (5 * R0)
