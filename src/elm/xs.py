from .elm import calculate_parameters
from .model_form import coulomb_charged_sphere, central, central_plus_coulomb, spin_orbit
from jitr import xs


def calculate_chex_ias_differential_xs(
    workspace: xs.quasielastic_pn.Workspace,
    *params,
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
        tuple(rxn.projectile),
        tuple(rxn.target),
        workspace.kinematics_entrance.Ecm,
        rxn.target.Efp,
        *params,
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
    *params,
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
        tuple(rxn.projectile),
        tuple(rxn.target),
        kinematics.Ecm,
        rxn.Ef,
        *params,
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
