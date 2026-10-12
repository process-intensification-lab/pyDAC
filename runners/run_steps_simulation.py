"""Dual CLI/VS Code launcher for a standalone TVSA step sequence."""
# PYTHON_ARGCOMPLETE_OK

from __future__ import annotations

import sys

from pydac.cli import LauncherSettings, main as pydac_main


# ---------------------------------------------------------------------------
# HOW THIS FILE WORKS
#   VS Code  : edit DEVELOPMENT_SETTINGS below, then run this file.
#   Terminal : python runners/run_steps_simulation.py --config preset.json \
#                     --set grid.nz=100 --dry-run     (same as `pydac steps ...`)
#   Development mode: Python defaults < CONFIG_PATH/--config < DEVELOPMENT_SETTINGS < --set.
#   Frozen mode: set CASE_PATH to a complete document; no settings are applied.
#   Resume   : set RESUME_PATH (or --resume) to a saved run folder; it reuses the
#              saved configuration and --set may change only operational settings.
#   Guide: docs/manual/01_inputs.md
# ---------------------------------------------------------------------------

# Set CASE_PATH to run one complete frozen document exactly. When set, CONFIG_PATH,
# DEVELOPMENT_SETTINGS, --config, and --set are all rejected (frozen mode is exact).
CASE_PATH = None

# Set CONFIG_PATH to start development mode from a partial or complete JSON
# document instead of the Python defaults (typically a case you already
# exported with --export-case and want to keep tweaking from VS Code, without
# hand-editing the JSON.) 
# DEVELOPMENT_SETTINGS below still applies on top, and
# a terminal --config overrides this constant when one is passed explicitly.
CONFIG_PATH = None

# Set RESUME_PATH to a saved run folder to continue or replay it in place
# (same as --resume). CONFIG_PATH and CASE_PATH must then be None.
RESUME_PATH = None

# OVERWRITE = True deletes this case's existing run folder (and only it) before a
# fresh run (same as --overwrite). A fresh run never writes into an occupied folder.
OVERWRITE = False

# DEVELOPMENT_SETTINGS overrides the default configuration of the model.
# Common hints:
#   grid.*       -> grid.nz, grid.nr, etc.
#   solver.*     -> solver.rtol, solver.atol, solver.atol_scalar_mass, etc.
#   geometry.*   -> geometry.aspect_ratio, geometry.volume_m3, etc.
#   bed.*        -> bed or bed.design.*, etc.
#   wall.*       -> wall.thickness_m, wall.overall_htc_override_W_m2K, etc.
#   sorbent.*    -> sorbent.particle_radius_m, sorbent.porosity, etc.
#   mixture.*    -> mixture.species, mixture.closure_species, mixture.adsorbates
#   adsorption.* -> named equilibrium_systems and species-keyed kinetics
#   steps.*      -> steps.ads.horizon.nominal_time_s, etc.
#                   steps.s_des.pressure.outlet_pressure_Pa, etc.
#                   steps.s_des.flow.steam_mass_flow_kg_s etc.
#   execution.*  -> execution.cycle_steps, execution.extra_tag, etc.
#   Full live catalog: python runners/run_steps_simulation.py --show-settings
#   User guide: docs/manual/01_inputs.md

packed_bed_design = {
    "architecture": "packed_bed",
    "design": {"eps_b": 0.38},
    "material": None,
}
column_design = packed_bed_design

DEVELOPMENT_SETTINGS = {}


def main(argv=None):
    """Run `pydac steps` with this file's constants (the terminal flags still apply)."""
    return pydac_main(
        ["steps", *(sys.argv[1:] if argv is None else argv)],
        launcher=LauncherSettings(
            case_path=CASE_PATH,
            config_path=CONFIG_PATH,
            resume_path=RESUME_PATH,
            development_settings=DEVELOPMENT_SETTINGS,
            overwrite=OVERWRITE,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
