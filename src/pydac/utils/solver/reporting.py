"""Terminal reporting for one CVODE solve outcome."""


def report_solve_outcome(
    sol_raw,
    termination_summary,
    duration,
    *,
    step,
    t_end_max,
):
    """Print consolidated CVODE diagnostics and termination information."""
    if not sol_raw.success:
        last_t = sol_raw.t[-1] if sol_raw.t is not None and len(sol_raw.t) else None
        print(f"[CVODE] Failed: status={sol_raw.status}, message='{sol_raw.message}'")
        if last_t is not None:
            print(f"[CVODE] Last t = {last_t:.6e}")

    print("-" * 60)
    print(f"Solve complete in {duration:.2f} seconds".center(60))
    print(
        f"CVODE stats: nfev={getattr(sol_raw, 'nfev', 'n/a')}, "
        f"njev={getattr(sol_raw, 'njev', 'n/a')}"
    )

    reason = termination_summary["reason"]
    if reason == "event" or termination_summary.get("event") is not None:
        event_summary = termination_summary["event"]
        prefix = "Failure event" if not termination_summary["success"] else "Termination reason"
        print(
            f"{prefix}: event '{event_summary['label']}' "
            f"at {event_summary['t_event_s']:.2f} s "
            f"({event_summary['metric_label']} = {event_summary['metric_value_str']}, "
            f"target {event_summary['threshold_str']})"
        )
    elif reason == "final_time":
        print(
            f"Termination reason: final time reached "
            f"at {termination_summary['t_final_s']:.2f} s"
        )
    elif reason == "event_not_reached":
        horizon_label = "t_end_max" if t_end_max is not None else "t_end"
        message = (
            f"Event mode requested for step '{step}', but no event was reached before "
            f"{horizon_label} ({termination_summary['t_final_s']:.2f} s)."
        )
        print(f"Termination reason: {message}")
        print("Continuing with the final state at the event-mode maximum time.")
    else:
        print(f"Termination reason: solver failure ({termination_summary['status']})")
    print("-" * 60 + "\n")


def report_saved_solve_outcome(solver_info):
    """Print diagnostics persisted with a solution during post-processing."""
    if not isinstance(solver_info, dict):
        return
    print("-" * 60)
    print("Saved CVODE solve outcome".center(60))
    print(f"CVODE success: {solver_info.get('success', 'n/a')}")
    print(f"CVODE status:  {solver_info.get('status', 'n/a')}")
    print(f"CVODE message: {solver_info.get('message', 'n/a')}")
    print(
        f"CVODE stats: nfev={solver_info.get('nfev', 'n/a')}, "
        f"njev={solver_info.get('njev', 'n/a')}"
    )
    print("-" * 60 + "\n")


def report_initial_event_skip(*, failure=False):
    """Report that an already-satisfied event bypassed CVODE integration."""
    event_kind = "failure event" if failure else "terminal event"
    print(f"Initial {event_kind} already satisfied at t=0; skipping CVODE integration.")
    print("CVODE stats: nfev=0, njev=0")


__all__ = [
    "note_once",
    "report_initial_event_skip",
    "report_saved_solve_outcome",
    "report_solve_outcome",
    "reset_notes",
]
