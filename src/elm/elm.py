import numpy as np

from rxmc.params import Parameter
from .model_form import coulomb_correction

params = [
    Parameter("V0", np.float64, r"MeV", r"V_0"),
    Parameter("W0", np.float64, r"MeV", r"W_0"),
    Parameter("Wd0", np.float64, r"MeV", r"W_{D0}"),
    Parameter("V1", np.float64, r"MeV", r"V_1"),
    Parameter("W1", np.float64, r"MeV", r"W_1"),
    Parameter("Wd1", np.float64, r"MeV", r"W_{D1}"),
    #   Parameter("eta", np.float64, r"dimensionless", r"\eta"),
    Parameter("alpha", np.float64, r"no-dim", r"\alpha"),
    Parameter("beta", np.float64, r"1/MeV", r"\beta"),
    Parameter("gamma_w", np.float64, r"MeV", r"\gamma_W"),
    Parameter("gamma_d", np.float64, r"MeV", r"\gamma_D"),
    # Parameter("r0", np.float64, r"fm", r"r_0"),
    # Parameter("r1", np.float64, r"fm", r"r_1"),
    Parameter("r0A", np.float64, r"fm", r"r_{0A}"),
    Parameter("r1A", np.float64, r"fm", r"r_{1A}"),
    Parameter("a0", np.float64, r"fm", r"a_0"),
    Parameter("a1", np.float64, r"fm", r"a_1"),
]
params_dtype = [(p.name, p.dtype) for p in params]
NUM_PARAMS = len(params)


def calculate_parameters(
    projectile: tuple,
    target: tuple,
    E: float,
    Ef: float,
    V0: float,
    W0: float,
    Wd0: float,
    V1: float,
    W1: float,
    Wd1: float,
    alpha: float,
    beta: float,
    gamma_w: float,
    gamma_d: float,
    r0A: float,
    r1A: float,
    a0: float,
    a1: float,
):
    r"""Calculate the parameters in the ELM for a given target isotope
    and energy, given a subparameter sample
    Parameters:
        projectile (tuple): projectile A,Z - must be neutron or proton ((1,0) or (1,1))
        target (tuple): target A,Z
        E (float): center-of-mass frame energy
        Ef  (float): Fermi energy for A,Z nucleus
        V0 (float): central isoscalar depth
        W0 (float): central isoscalar surface depth
        Wd0 (float): central isoscalar derivative depth
        V1 (float): central isovector depth
        W1 (float): central isovector surface depth
        Wd1 (float): central isovector derivative depth
        alpha (float): 1st order energy dependence
        beta (float): 2nd order energy dependence
        gamma_w (float): energy dependence parameter for W
        gamma_d (float): energy dependence parameter for Wd
        r0A (float): radius parameter for central isoscalar potential
        r1A (float): radius parameter for central isovector potential
        a0 (float): surface diffuseness parameter for central isoscalar potential
        a1 (float): surface diffuseness parameter for central isovector potential
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
    #R0 = -0.2 + r0A * A ** (1.0 / 3.0)
    #R1 = -0.2 + r1A * A ** (1.0 / 3.0)
    R0 = r0A * A ** (1.0 / 3.0)
    R1 = r1A * A ** (1.0 / 3.0)

    # Coulomb radius equal to central isoscalar radius
    RC = R0

    # energy
    dE = E
    # dE = E - Ef
    if projectile == (1, 1):
        deltaVC = coulomb_correction(A, Z, RC)
        # dE -= deltaVC

    # energy dependence of depths
    erg_v = 1 + (alpha * dE + beta * dE**2) / V0
    erg_w = dE**2 / (dE**2 + gamma_w**2)
    # erg_wd = dE**2 / (dE**2 + gamma_d**2)
    erg_wd = dE**2 / (dE**2 + gamma_d**2)* np.exp(-dE / gamma_d)

    # central isoscalar depths
    V0 = V0 * erg_v
    W0 = W0 * erg_w
    Wd0 = Wd0 * erg_wd

    # central isovector depths
    V1 = V1 * erg_v
    W1 = W1 * erg_w
    Wd1 = Wd1 * erg_wd

    # spin orbit depths are just a fixed ratio (eta) of central depths
    # for now, keep this fixed
    # eta = 0.44  # eta

    # spin orbit isovector depths
    # Vso0 = V0 * eta
    # fix at KDUQ value but convert from form using
    # (hbar/mpi c)^2 * l.sigma to 1/r0^2 * (l.s)
    # Wso0 = W0 * eta

    # spin orbit isovector depths
    # Vso1 = V1 * eta
    # Wso1 = W1 * eta

    # alternative option just fixing all so depths to be A,E independent:
    Vso0 = 5.58
    Wso0 = 0
    Vso1 = 0
    Wso1 = 0

    return (
        (V0, W0, Wd0, R0, a0, R0, a0),
        (V1, W1, Wd1, R1, a1, R1, a1),
        (Vso0, Wso0, R0, a0),
        (Vso1, Wso1, R1, a1),
        (Z*Zp, RC),
        asym_factor,
    )
