"""
pyDAC - Aspect-ratio geometry conversions.

This utility module converts the chosen tau = L/D ratio and bed volume into
consistent column length and radius values. Reference geometry is read lazily
from the public experiment profile so this low-level utility does not own a
second copy of the defaults.

Author: Mattia Galanti
Affiliation: Eindhoven University of Technology
Email: mattiagalanti1@gmail.com
Last updated: 2026-04-10
"""

import numpy as np


def _reference_geometry():
    from pydac.config import default_experiment_config

    return default_experiment_config().simulation.geometry


def get_L_and_R(tau_ar=None, volume=None):
    """
    Compute bed length and radius from aspect ratio and volume.

    Parameters
    ----------
    tau_ar : float or None
        Column aspect ratio defined as L/D. If omitted, the reference profile
        supplies it.
    volume : float or None
        Bed volume [m^3]. If omitted, the reference profile supplies it.

    Returns
    -------
    tuple[float, float]
        `(L_b, R_b)` in meters.
    """
    reference = _reference_geometry() if tau_ar is None or volume is None else None
    tau_ar = reference.aspect_ratio if tau_ar is None else tau_ar
    volume = reference.volume_m3 if volume is None else volume

    tau_val = float(tau_ar)
    if tau_val <= 0.0:
        raise ValueError(f"tau_ar must be > 0. Received: {tau_ar}")

    vol = float(volume)
    if vol <= 0.0:
        raise ValueError(f"volume must be > 0. Received: {volume}")

    diameter = (4.0 * vol / (np.pi * tau_val)) ** (1.0 / 3.0)
    length = tau_val * diameter
    radius = diameter / 2.0
    return length, radius
