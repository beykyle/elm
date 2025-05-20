from collections import OrderedDict
from json import load, dumps
from pathlib import Path

import pandas as pd
import numpy as np

from jitr.optical_potentials.potential_forms import (
    thomas_safe,
    woods_saxon_prime_safe,
    woods_saxon_safe,
    coulomb_charged_sphere,
)
from jitr.utils.constants import MASS_PION, ALPHA, HBARC
from jitr.xs.elastic import DifferentialWorkspace


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
    return [OrderedDict(zip([p.name for p in params], entry)) for entry in samples]


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


def isoscalar(r, V0, W0, Wd0, R0, a0, Rd0, ad0):
    r"""isoscalar part (without spin-orbit) as a function of radial distance r"""
    volume = -(V0 + 0j * W0) * woods_saxon_safe(r, R0, a0)
    surface = (4j * a0 * Wd0) * woods_saxon_prime_safe(r, Rd0, ad0)
    return volume + surface


def isovector(r, V1, W1, Wd1, R1, a1, Rd1, ad1):
    r"""isovector part (without spin-orbit) as a function of radial distance r"""
    volume = -(V1 + 1j * W1) * woods_saxon_safe(r, R1, a1)
    surface = (4j * a1 * Wd1) * woods_saxon_prime_safe(r, Rd1, ad1)
    return volume + surface


def spin_orbit(r, Vso, R0, a0):
    r"""spin-orbit term"""
    return Vso / MASS_PION**2 * thomas_safe(r, R0, a0)


def central_plus_coulomb(
    r, asym_factor, isoscalar_params, isovector_params, coulomb_params
):
    r"""sum of coulomb, isoscalar and isovector terms, without spin orbit"""
    coulomb = coulomb_charged_sphere(r, *coulomb_params)
    centr = central(r, asym_factor, isoscalar_params, isovector_params)
    return centr + coulomb


def central(
    r,
    asym_factor,
    isoscalar_params,
    isovector_params,
):
    r"""sum of isoscalar and isovector terms, without spin orbit"""
    V0 = isoscalar(r, *isoscalar_params)
    V1 = isovector(r, *isovector_params)
    return V0 + asym_factor * V1


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
        isoscalar_params (tuple): (V0, W0, Wd0, R0, a0, Rd, ad)
        isovector_params (tuple): (V1, W1, Wd1, R1, a1, Rd1, ad1)
        spin_orbit_params (tuple): (Vso, rso, aso)
        Coulomb_params (tuple): (Zz, Rc)
        asym_factor =  +(-) (N-Z)/(N+Z), for neutrons(protons)
    """
    # asymmetry for isovector dependence
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

    # Coulomb radius equal to isoscalar radius
    RC = R0

    # energy
    dE = E - Ef
    if projectile == (1, 1):
        dE -= coulomb_correction(A, Z, RC)

    # energy dependence of depths
    erg_v = 1.0 + params["alpha"] * dE  # + params["beta"] * dE**2
    erg_w = dE**2 / (dE**2 + params["gamma_w"] ** 2)
    erg_wd = dE**2 / (dE**2 + params["gamma_d"] ** 2)

    # isoscalar depths
    V0 = params["V0"] * erg_v
    W0 = params["W0"] * erg_w
    Wd0 = params["Wd0"] * erg_wd
    Vso = 5.58  # params["Vso"]

    # isovector depths
    V1 = params["V1"] * erg_v
    W1 = params["W1"] * erg_w
    Wd1 = params["Wd1"] * erg_wd

    return (
        (V0, W0, Wd0, R0, a0, R0, a0),
        (V1, W1, Wd1, R1, a1, R1, a1),
        (Vso, R0, a0),
        (Z, RC),
        asym_factor,
    )


def coulomb_correction(A, Z, RC):
    r"""
    Coulomb correction for proton energy
    """
    return 6.0 * Z * ALPHA * HBARC / (5 * RC)


def calculate_diff_xs(
    workspace: DifferentialWorkspace,
    params: OrderedDict,
):
    rxn = workspace.reaction
    kinematics = workspace.kinematics
    assert rxn.projectile.A == 1
    (
        isoscalar_params,
        isovector_params,
        spin_orbit_params,
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
            isoscalar_params,
            isovector_params,
            coul_params,
        ),
        args_spin_orbit=spin_orbit_params,
    )
