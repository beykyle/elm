from collections import OrderedDict
from json import load, dumps
from pathlib import Path

import pandas as pd
import numpy as np

from jitr.reactions.potentials import (
    thomas_safe,
    woods_saxon_prime_safe,
    woods_saxon_safe,
    coulomb_charged_sphere,
)
from jitr.utils.constants import MASS_PION, ALPHA, HBARC


class Parameter:
    def __init__(self, name, dtype, unit, latex_name):
        self.name = name
        self.dtype = dtype
        self.unit = unit
        self.latex_name = latex_name


params = [
    Parameter("V0", np.float64, r"MeV", r"V_0"),
    Parameter("W0", np.float64, r"MeV", r"W_0"),
    Parameter("Wd0", np.float64, r"MeV", r"W_{D0}"),
    Parameter("V1", np.float64, r"MeV", r"V_1"),
    Parameter("W1", np.float64, r"MeV", r"W_1"),
    Parameter("Wd1", np.float64, r"MeV", r"W_{D1}"),
    #   Parameter("Vso", np.float64, r"MeV", r"V_{so}"),
    Parameter("alpha", np.float64, r"MeV$^{-1}$", r"\alpha"),
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


def dump_sample_to_json(fpath: Path, sample: OrderedDict):
    with open(fpath, "w") as file:
        file.write(dumps(dict(sample), indent=4))


def read_sample_from_json(fpath: Path):
    try:
        with open(fpath, "r") as file:
            return load(file, object_pairs_hook=OrderedDict)
    except IOError:
        raise f"Error: failed to open {fpath}"


def array_to_list(samples: np.ndarray):
    return [
        OrderedDict([(key, entry[key]) for key in params_dtype.names])
        for entry in samples
    ]


def list_to_array(samples: list):
    return np.array([(sample.values()) for sample in samples], dtype=params_dtype)


def list_to_dataframe(samples: list):
    return pd.DataFrame(samples)


def dataframe_to_list(samples: pd.DataFrame):
    return samples.to_dict(orient="records", into=OrderedDict)


def dump_samples_to_numpy(fpath: Path, samples: list):
    list_to_array(samples).save(fpath)


def read_samples_from_numpy(fpath: Path):
    return array_to_list(np.load(fpath))


def isoscalar(r, V0, W0, Wd0, R0, a0, Rd, ad):
    r"""isoscalar part (without spin-orbit) as a function of radial distance r"""
    return -(V0 + 1j * W0) * woods_saxon_safe(r, R0, a0) + (
        4j * a0 * Wd0
    ) * woods_saxon_prime_safe(r, Rd, ad)


def isovector(r, V1, W1, Wd1, R1, a1, Rd1, ad1):
    r"""isovector part (without spin-orbit) as a function of radial distance r"""
    return -(V1 + 1j * W1) * woods_saxon_safe(r, R1, a1) + (
        4j * a1 * Wd1
    ) * woods_saxon_prime_safe(r, Rd1, ad1)


def spin_orbit(r, Vso, R0, a0):
    r"""spin-orbit term"""
    return Vso / MASS_PION**2 * thomas_safe(r, R0, a0)


def central_plus_coulomb(
    r, projectile, asym_factor, isoscalar_params, isovector_params, coulomb_params
):
    r"""sum of coulomb, isoscalar and isovector terms, without spin orbit"""
    if projectile == (1, 1):
        coul = coulomb_charged_sphere(r, *coulomb_params)
    elif projectile == (1, 0):
        asym_factor *= -1
        coul = 0

    nucl = isoscalar(r, *isoscalar_params) + asym_factor * isovector(
        r, *isovector_params
    )
    return nucl + coul


def central(
    r,
    projectile,
    asym_factor,
    isoscalar_params,
    isovector_params,
):
    r"""sum of isoscalar and isovector terms, without spin orbit"""
    if projectile == (1, 0):
        asym_factor *= -1

    nucl = isoscalar(r, *isoscalar_params) + asym_factor * isovector(
        r, *isovector_params
    )
    return nucl


def calculate_parameters(
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
        isoscalar_params (tuple)
        isovector_params (tuple)
        spin_orbit_params (tuple)
        Coulomb_params (tuple)
        asym_factor =  (N-Z)/(N+Z)
    """
    # asymmetry for isovector dependence
    A, Z = target
    asym_factor = (A - 2 * Z) / (A)

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
        dE -= coulomb_correction(A, Z, RC)

    # energy dependence of depths
    erg_v = 1.0 + sp["alpha"] * dE  # + sp["beta"] * dE**2
    erg_w = dE**2 / (dE**2 + sp["gamma_w"] ** 2)
    erg_wd = dE**2 / (dE**2 + sp["gamma_d"] ** 2)

    # isoscalar depths
    V0 = sp["V0"] * erg_v
    W0 = sp["W0"] * erg_w
    Wd0 = sp["Wd0"] * erg_wd
    Vso = 5.58  # sp["Vso"]

    # isovector depths
    V1 = sp["V1"] * erg_v
    W1 = sp["W1"] * erg_w
    Wd1 = sp["Wd1"] * erg_wd

    return (
        (V0, W0, Wd0, R0, a0),
        (V1, W1, Wd1, R1, a1),
        (Vso, R0, a0),
        (Z, RC),
        asym_factor,
    )


def coulomb_correction(A, Z, RC):
    r"""
    Coulomb correction for proton energy
    """
    return 6.0 * Z * ALPHA * HBARC / (5 * RC)
