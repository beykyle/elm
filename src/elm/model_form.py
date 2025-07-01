from jitr.optical_potentials.potential_forms import (
    thomas_safe,
    woods_saxon_prime_safe,
    woods_saxon_safe,
    coulomb_charged_sphere,
)
from jitr.utils.constants import ALPHA, HBARC, WAVENUMBER_PION


def central_form(r, V, W, Wd, R, a, Rd, ad):
    r"""form of central part (volume and surface) as a function of radial distance"""
    volume = -(V + 1j * W) * woods_saxon_safe(r, R, a)
    surface = (4j * ad * Wd) * woods_saxon_prime_safe(r, Rd, ad)
    return volume + surface


def spin_orbit_form(r, Vso, Wso, R, a):
    r"""form of spin-orbit term"""
    return (Vso + 1j * Wso) / WAVENUMBER_PION ** 2 * thomas_safe(r, R, a)


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
