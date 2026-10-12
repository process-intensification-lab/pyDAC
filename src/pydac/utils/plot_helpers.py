"""
pyDAC - Manual plotting overrides for profile figures.

This module stores curated time-window and snapshot selections used by the
profile-plotting routines to keep 2D and outlet figures focused on the most
informative parts of each TVSA step.

Author: Mattia Galanti
Affiliation: Eindhoven University of Technology
Email: mattiagalanti1@gmail.com
Last updated: 2026-04-10
"""

T_END_OVERRIDES = {
    "startup": {
        "y_CO2": 12 * 3600,
        "y_O2": 12 * 3600,
        "y_N2": 12 * 3600,
        "q_CO2": 12 * 3600,
        "y_H2O": 1 * 3600,
        "q_H2O": 1 * 3600,
        "T_bed": 1 * 3600,
        "p_tot": 5 * 3600,
    },
    
    "startup": {
        "y_CO2": 12 * 3600,
        "y_O2": 12 * 3600,
        "y_N2": 12 * 3600,
        "q_CO2": 12 * 3600,
        "y_H2O": 1 * 3600,
        "q_H2O": 1 * 3600,
        "T_bed": 1 * 3600,
        "p_tot": 5 * 3600,
    },
        
    "blow": {
        "y_CO2": None,
        "y_O2": None,
        "y_N2": None,
        "q_CO2": None,
        "y_H2O": None,
        "q_H2O": None,
        "T_bed": None,
        "p_tot": None,
    },
    
    "heat": {
        "y_CO2": None,
        "y_O2": None,
        "y_N2": None,
        "q_CO2": None,
        "y_H2O": None,
        "q_H2O": None,
        "T_bed": None,
        "p_tot": None,
    },

    "w_purge": {
        "y_CO2": None,
        "y_O2": None,
        "y_N2": None,
        "q_CO2": None,
        "y_H2O": None,
        "q_H2O": None,
        "T_bed": None,
        "p_tot": None,
    },
    
    "w_des": {
        "y_CO2": 2 * 3600,
        "y_O2": 2 * 3600,
        "y_N2": 2 * 3600,
        "q_CO2": 10 * 3600,
        "y_H2O": 3 * 3600,
        "q_H2O": 3 * 3600,
        "T_bed": 2 * 3600,
        "p_tot": None,
    },
    
    "s_des": {
        "y_CO2": None,
        "y_O2": None,
        "y_N2": None,
        "q_CO2": None,
        "y_H2O": None,
        "q_H2O": None,
        "T_bed": None,
        "p_tot": None,
    },
}

# Optional exact snapshot times for 2D plots.
# If a field is listed here, these times take precedence over the uniform
# sampling based on `T_END_OVERRIDES`.
#
# Example:
# "startup": {
#     "y_CO2": {"times": (0.5, 1, 5, 10, 30, 50), "unit": "min"},
# }
SNAPSHOT_TIME_OVERRIDES = {
    "startup": {
        "y_H2O": {"times": (1, 3, 5, 10, 30, 60), "unit": "min"},
        "q_H2O": {"times": (5, 10, 30, 60, 120, 240), "unit": "min"},
        "T_bed": {"times": (1, 3, 5, 10, 25, 60), "unit": "min"},
        "y_O2": {"times": (1, 5, 10, 30, 60, 120), "unit": "min"},
        "y_N2": {"times": (1, 5, 10, 30, 60, 120), "unit": "min"},
    },
    "s_des": {
        "y_H2O": {"times": (1, 3, 5, 10, 20, 60), "unit": "min"},
        "q_H2O": {"times": (1, 3, 5, 10, 20, 60), "unit": "min"},
        "T_bed": {"times": (1, 3, 5, 10, 25, 60), "unit": "min"},
        "q_CO2": {"times": (1, 5, 10, 20, 30, 60), "unit": "min"},
        "y_CO2": {"times": (1, 3, 5, 10, 30, 60), "unit": "min"},
        "y_O2": {"times": (0, 3, 5, 10, 30, 60), "unit": "min"},
        "y_N2": {"times": (0, 3, 5, 10, 30, 60), "unit": "min"},
        
    },
    "ads": {
        "y_H2O": {"times": (1, 3, 5, 10, 30, 60), "unit": "min"},
        "q_H2O": {"times": (5, 10, 30, 60, 120, 240), "unit": "min"},
        "T_bed": {"times": (1, 3, 5, 10, 25, 60), "unit": "min"},
        "y_O2": {"times": (1, 5, 10, 30, 60, 120), "unit": "min"},
        "y_N2": {"times": (1, 5, 10, 30, 60, 120), "unit": "min"},
    },
}
