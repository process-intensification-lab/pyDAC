"""Outlet-face histories evaluated with the initialized model's hydraulic law.

These are derived observations, not extra PDE states. Replay before dimensional
scaling/clipping so the concentrations and density match the RHS exactly.
"""

import numpy as np


def outlet_face_history(model, packed_history):
    """Return signed superficial velocity, species molar flows and sensible
    enthalpy flow at z=L.

    Velocity has shape (Nr, Nt); each species flow and the enthalpy flow have
    shape (Nr, Nt) and include the annular area. Positive flow leaves the
    column. The selected pressure law is already bound on the model,
    independently of case inputs.
    """
    nt = packed_history.shape[1]
    velocity = np.zeros((model.Nr, nt))
    flows = {species: np.zeros_like(velocity) for species in model.gas_species}
    enthalpy_flow = np.zeros_like(velocity)
    if model.step not in ("press", "cool"):
        last, inner = model.outlet_face_weights
        for k in range(nt):
            state = packed_history[:, k]
            c_tot = state[model.state_field_by_name["c_tot"].state_slice]
            temperature = state[model.state_field_by_name["T_bed"].state_slice]
            c_tot_dim = c_tot * model.c_tot_ref
            _, _, fractions = model._reconstruct_gas_state(state, c_tot_dim, 1e-32)
            density = c_tot_dim * (fractions @ model.MW)
            u = model._velocity_from_pressure_law(c_tot * temperature, density)
            velocity[:, k] = u[-1, :, 0] * model.u_ref
            concentrations = model._c_full_dim_buf
            face = last * concentrations[-1] + inner * concentrations[model._outlet_face_inner_index]
            for species, index in model.gas_index.items():
                flows[species][:, k] = face[:, index] * velocity[:, k] * model.A_rings
            # The energy balance transports `c_tot_dim * h_g` as a single
            # scalar, so the face value has to interpolate that product.
            # Interpolating temperature and concentrations apart does not
            # return it, and the cross term is exactly the radial
            # non-uniformity this reporting exists to resolve.
            sensible = (
                c_tot_dim
                * (fractions @ model.cp_g)
                * (temperature - model.T_enthalpy_ref_nd)
                * model.T_ref
            ).reshape(model.Nz, model.Nr)
            sensible_face = (
                last * sensible[-1] + inner * sensible[model._outlet_face_inner_index]
            )
            enthalpy_flow[:, k] = sensible_face * velocity[:, k] * model.A_rings
    return {
        "source": "solver_outlet_face",
        "velocity_m_s": velocity,
        "species_molar_flow_mol_s": flows,
        "sensible_enthalpy_flow_W": enthalpy_flow,
    }
