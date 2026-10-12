"""Final-cycle plotting reconstruction for CSS runs."""
from pydac.utils.css.cycle import (
    _cycle_folder_is_spatial_root,
    _load_previous_cycle_terminal_fields,
    _resolve_cycle_folder,
)
from pydac.utils.tvsa import build_cyclic_cycle_steps, normalize_step_options

def plot_cycle_at_css(
    base_folder,
    css_cycle_idx,
    prev_fields=None,
    cycle_steps=None,
    desorption_step="w_des",
    termination_mode=None,
    step_options=None,
    print_to_terminal=True,
    cycle_config=None,
    persist_step_outputs=True,
):
    """Replay the saved CSS cycle ``css_cycle_idx`` (no solve) and draw its step plots.

    ``persist_step_outputs=False`` (``pydac.report``) draws the plots only: the step logs,
    step metrics and their provenance stay as the run wrote them.
    """
    from pydac.core.TVSA_cycle import TVSA_Cycle as AdsorptionCycle
    del desorption_step  # Cycle-order resolution now comes from the shared cyclic builder.
    if cycle_config is None:
        raise TypeError("`cycle_config` is required to reconstruct CSS profiles.")

    # Reconstruct the cycle from the same resolved runtime config used to solve it.
    cycle_folder = _resolve_cycle_folder(base_folder, css_cycle_idx)
    adsorption_cycle = AdsorptionCycle(
        cycle_config,
        base_folder=cycle_folder,
        cyclic_mode=True,
        cycle_idx=css_cycle_idx,
        last_cycle_idx=css_cycle_idx,
        print_to_terminal=print_to_terminal,
        base_folder_is_spatial_root=_cycle_folder_is_spatial_root(cycle_folder),
    )

    steps = build_cyclic_cycle_steps(css_cycle_idx, cycle_steps=cycle_steps)
    normalized_step_options = normalize_step_options(step_options)

    if prev_fields is None:
        prev_fields = _load_previous_cycle_terminal_fields(
            adsorption_cycle=adsorption_cycle,
            base_folder=base_folder,
            css_cycle_idx=css_cycle_idx,
            cycle_steps=cycle_steps,
        )

    for step_name in steps:
        step_termination_mode = normalized_step_options.get(step_name, {}).get(
            "termination_mode",
            termination_mode,
        )
        kwargs = {
            "step_name": step_name,
            "solve": False,
            "plot_profiles": True,
            "persist_step_outputs": persist_step_outputs,
        }
        if step_termination_mode is not None:
            kwargs["termination_mode"] = step_termination_mode
        if prev_fields is not None:
            kwargs["prev_fields"] = prev_fields

        final_fields, _ = adsorption_cycle._run_step(**kwargs)
        prev_fields = final_fields
