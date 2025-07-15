from .elm import calculate_parameters


def elm_elastic(ws, *x):
    r"""
    Callable for
    `rxmc.ElasticDifferentialXSModel.calculate_interaction_from_params`

    Given a workspace and a set of ELM parameters, returns the args for
    `elm.model_form.central_plus_coulomb` and `elm.model_form.spin_orbit`

    Parameters
    ----------
    ws : Workspace
        The workspace containing the reaction and kinematics.
    x : tuple
        The parameters for the ELM model.
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
    ) = calculate_parameters(
        tuple(rxn.projectile), tuple(rxn.target), kinematics.Ecm, rxn.Ef, *x
    )
    args_central = (
        asym_factor,
        central_isoscalar_params,
        central_isovector_params,
        coul_params,
    )
    args_spin_orbit = (
        asym_factor,
        spin_orbit_isoscalar_params,
        spin_orbit_isovector_params,
    )
    return args_central, args_spin_orbit
