"""Campaign paths and typed grid configuration helpers."""

import hashlib
import json

from pydac.utils.run.validation import normalize_termination_mode


def resolve_grid_config(grid_config=None, *, nz=None, nr=None):
    """Return the public grid config, optionally replacing dimensions."""
    from pydac.config import GridConfig, default_experiment_config

    base = grid_config if grid_config is not None else default_experiment_config().simulation.grid
    if not isinstance(base, GridConfig):
        raise TypeError("`grid_config` must be a GridConfig instance or None.")
    updates = {name: value for name, value in (("nz", nz), ("nr", nr)) if value is not None}
    return base.with_overrides(**updates)


def _short_simulation_hash(payload) -> str:
    rendered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(rendered.encode("ascii")).hexdigest()[:8]


def _legacy_simulation_hash(
    *,
    cycle,
    cycle_steps,
    desorption_step,
    co_ads,
    tau_ar,
    termination_mode,
    mode_override,
    bed_config,
) -> str:
    """Hash the old explicit arguments for callers without a typed case."""
    payload = {
        "cycle": cycle,
        "cycle_steps": list(cycle_steps) if cycle_steps is not None else None,
        "desorption_step": desorption_step,
        "co_ads": co_ads,
        "tau_ar": tau_ar,
        "termination_mode": termination_mode,
        "mode_override": mode_override,
        "bed_config": bed_config,
    }
    return _short_simulation_hash(payload)


def _column_type(experiment_config=None, bed_config=None) -> str:
    if experiment_config is not None:
        return experiment_config.simulation.bed.architecture

    from pydac.core.bed_architectures import resolve_bed_config

    return resolve_bed_config(bed_config)["architecture"]


def make_base_folder(
    cycle: bool = False,
    cycle_steps=None,
    desorption_step: str = "w_des",
    co_ads: bool = False,
    tau_ar: float | None = None,
    termination_mode=None,
    extra_tag: str | None = None,
    mode_override: str | None = None,
    bed_config=None,
    output_root: str | None = None,
    experiment_config=None,
    identity_hash: str | None = None,
) -> str:

    """
    Build the default results folder path.

    Parameters
    ----------
    extra_tag : str | None, optional
        Optional custom prefix for the final folder name. If provided, it is
        inserted at the beginning of the leaf folder token (e.g.
        ``TEST_packed_bed_2145bdd7``). If None or empty, the leaf starts with
        the column type and contains only the simulation hash after it.
    mode_override : str | None, optional
        If provided, replaces the computed ``{mode}`` segment of the path
        (normally ``cyclic_mode_time``, ``step_mode``, etc.). Used by
        campaigns like NSGA-II where the cyclic/termination distinction is
        fixed and the segment should carry the campaign name instead.
    bed_config : mapping or None, optional
        Retained for compatibility with callers that have not migrated to a
        typed experiment. It contributes to the compatibility hash when
        ``experiment_config`` is omitted.
    experiment_config : ExperimentConfig or None, optional
        Complete resolved experiment. When provided, the leaf hash is the
        first eight characters of the same case fingerprint written to
        ``case_metadata.json``. This is the preferred path because it makes
        the folder identity unique to the complete simulation configuration.
    identity_hash : str | None, optional
        A campaign identity that extends the case, such as an NSGA-II
        campaign's search space and optimizer. When given, its first eight
        characters replace the case fingerprint in the leaf.
    """

    if identity_hash is not None:
        simulation_hash = identity_hash[:8]
    elif experiment_config is not None:
        from pydac.inputs.case_io import case_fingerprint

        simulation_hash = case_fingerprint(experiment_config)[:8]
    else:
        simulation_hash = _legacy_simulation_hash(
            cycle=cycle,
            cycle_steps=cycle_steps,
            desorption_step=desorption_step,
            co_ads=co_ads,
            tau_ar=tau_ar,
            termination_mode=termination_mode,
            mode_override=mode_override,
            bed_config=bed_config,
        )
    column_type = _column_type(experiment_config, bed_config)

    if output_root is None:
        from pydac.config import default_experiment_config

        reference = default_experiment_config()
        output_root = (
            experiment_config.execution.output_root
            if experiment_config is not None
            else reference.execution.output_root
        )

    if mode_override is not None:
        mode = mode_override
    else:
        # Choose mode
        mode = "cyclic_mode" if cycle else "step_mode"
        normalized_termination_mode = normalize_termination_mode(
            termination_mode,
            allow_none=True,
        )
        if normalized_termination_mode is not None:
            mode += f"_{normalized_termination_mode}"

    # Optional custom prefix for the leaf folder name.
    tag_prefix = ""
    if extra_tag is not None:
        if not isinstance(extra_tag, str):
            raise TypeError(
                f"`extra_tag` must be str or None. Received type: {type(extra_tag).__name__}."
            )
        cleaned_tag = extra_tag.strip().replace(" ", "_")
        if cleaned_tag:
            tag_prefix = f"{cleaned_tag}_"

    # Keep only the architecture label human-readable; the complete simulation
    # identity lives in the hash and in the metadata.
    leaf = f"{tag_prefix}{column_type}_{simulation_hash}"

    base = f"{output_root}/{mode}/{leaf}"

    return base
