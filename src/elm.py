from collections import OrderedDict
from json import load, dumps
from pathlib import Path

import numpy as np

import jitr
from jitr.reactions.potentials import (
    thomas_safe,
    woods_saxon_prime_safe,
    woods_saxon_safe,
)
from jitr.utils.constants import MASS_PION, ALPHA, HBARC


class Parameter:
    def __init__(self, name, dtype, fancy_label):
        self.name = name
        self.dtype = dtype
        self.fancy_label = fancy_label


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
    Parameter("eta_d", np.float64, r"$\eta_D$ [MeV]"),
    Parameter("r0", np.float64, r"$r_0$ [fm]"),
    Parameter("r1", np.float64, r"$r_1$ [fm]"),
    Parameter("r0A", np.float64, r"$r_{0A}$ [fm]"),
    Parameter("r1A", np.float64, r"$r_{1A}$ [fm]"),
    Parameter("a0", np.float64, r"$a_0$ [fm]"),
    Parameter("a1", np.float64, r"$a_1$ [fm]"),
    Parameter("ad0", np.float64, r"$a_{d0}$ [fm]"),
    Parameter("ad1", np.float64, r"$a_{d1}$ [fm]"),
]
elm_params_dtype = [(p.name, p.dtype) for p in elm_params]
elm_fancy_labels = dict([(p.name, p.fancy_label) for p in elm_params])
N_PARAMS = len(elm_params)


def dump_sample_to_json(fpath: Path, sample: OrderedDict):
    pass


def dump_samples_to_json(fpath: Path, samples: list):
    pass


def read_sample_from_json(fpath: Path):
    pass


def read_samples_from_json(fpath: Path):
    pass


def to_list_of_samples(samples: list):
    return [
        OrderedDict([(key, entry[key]) for key in elm_params_dtype.names])
        for entry in samples
    ]


def to_numpy(samples: list):
    return np.array([(sample.values()) for sample in samples], dtype=elm_params_dtype)


def dump_samples_to_numpy(fpath: Path, samples: list):
    to_numpy(samples).save(fpath)


def read_samples_from_numpy(fpath: Path):
    return to_list_of_samples(np.load(fpath))


def elm_isoscalar(r, V0, W0, Wd0, R0, a0, Rd, ad):
    r"""isoscalar part of the EL model (without spin-orbit) as a function of radial distance r"""
    return -(V0 + 1j * W0) * woods_saxon_safe(r, R0, a0) + (
        4j * a0 * Wd0
    ) * woods_saxon_prime_safe(r, Rd, ad)


def elm_isovector(r, V1, W1, Wd1, R1, a1, Rd1, ad1):
    r"""isovector part of the EL model (without spin-orbit) as a function of radial distance r"""
    return -(V1 + 1j * W1) * woods_saxon_safe(r, R1, a1) + (
        4j * a1 * Wd1
    ) * woods_saxon_prime_safe(r, Rd1, ad1)


def elm_so(r, Vso, R0, a0):
    r"""ELM spin-orbit term"""
    return Vso / MASS_PION**2 * thomas_safe(r, R0, a0)


def elm_spin_scalar_plus_coulomb(
    r, projectile, alpha, isoscalar_params, isovector_params, coulomb_params
):
    if projectile == (1, 1):
        factor = 1
        coul = jitr.reactions.potentials.coulomb_charged_sphere(r, *coulomb_params)
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

    # -1 for neutrons +1 for protons
    _, Z = projectile
    factors = [-1, 1]
    factor = factors[Z]

    nucl = elm_isoscalar(r, *isoscalar_params) + factor * alpha * elm_isovector(
        r, *isovector_params
    )
    return nucl


def el_model_parameters(
    projectile: tuple, target: tuple, E: float, Ef: float, sp: OrderedDict
):
    r"""Calculate the parameters in the ELM for a given target isotope
    and energy, given a subparameter sample
    Parameters:
        A (int): target mass
        Z (int): target charge
        E (float): center-of-mass frame energy
        E  (float): Fermi energy for A,Z nucleus
        sp (OrderedDict) : subparameter sample
    Returns:
        isoscalar_sp (tuple)
        isovector_sp (tuple)
        spin_orbit_sp (tuple)
        Coulomb_sp (tuple)
        delta =  (N-Z)/(N+Z)
    """
    # asymmetry for isovector dependence
    A, Z = target
    delta = (A - 2 * Z) / (A)

    # geometries
    R0 = sp["r0"] + sp["r0A"] * A ** (1.0 / 3.0)
    R1 = sp["r1"] + sp["r1A"] * A ** (1.0 / 3.0)
    a0 = sp["a0"]
    a1 = sp["a1"]

    # Coulomb radius equal to isoscalar radius
    RC = R0

    # energy
    dE = E - Ef
    if projectile == (1, 1):
        dE -= elm_coulomb_correction(A, Z, RC)

    # energy dependence of depths
    erg_v = 1.0 + sp["alpha"] * dE + sp["beta"] * dE**2
    erg_w = dE**2 / (dE**2 + sp["gamma_w"] ** 2)
    erg_wd = dE**2 / (dE**2 + sp["gamma_d"] ** 2) * np.exp(-dE / sp["eta_d"])

    # isoscalar depths
    V0 = sp["V0"] * erg_v
    W0 = sp["W0"] * erg_w
    Wd0 = sp["Wd0"] * erg_wd
    Vso = sp["Vso"]

    # isovector depths
    V1 = sp["V1"] * erg_v
    W1 = sp["W1"] * erg_w
    Wd1 = sp["Wd1"] * erg_wd

    return (
        (V0, W0, Wd0, R0, a0),
        (V1, W1, Wd1, R1, a1),
        (Vso, R0, a0),
        (Z, RC),
        delta,
    )


def elm_coulomb_correction(A, Z, RC):
    r"""
    Coulomb correction for proton energy
    """
    return 6.0 * Z * ALPHA * HBARC / (5 * RC)
