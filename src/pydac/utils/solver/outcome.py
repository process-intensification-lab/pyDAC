"""Compact, JSON-safe handling of final CVODE failures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _plain_scalar(value: Any) -> Any:
    """Return a JSON-safe scalar without importing NumPy."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _plain_scalar(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def compact_solver_outcome(
    solution: Mapping[str, Any] | None,
    *,
    step: str,
) -> dict[str, Any]:
    """Return the small solver record carried by cycle/candidate metadata."""
    solution = solution if isinstance(solution, Mapping) else {}
    cvode = solution.get("cvode")
    cvode = cvode if isinstance(cvode, Mapping) else {}

    times = solution.get("t")
    final_time_s = None
    try:
        if times is not None and len(times):
            final_time_s = float(times[-1])
    except (IndexError, TypeError, ValueError):
        final_time_s = None

    missing_metadata = "success" not in cvode
    message = cvode.get("message")
    if missing_metadata:
        message = "Final CVODE success metadata is missing."

    termination = solution.get("termination")
    initial_event_skip = (
        isinstance(termination, Mapping)
        and termination.get("status") == "initial_event"
    )
    effective_solver = (
        "not_run"
        if initial_event_skip
        else cvode.get("linsolver", solution.get("solver"))
    )
    state_domain_failure = (
        isinstance(termination, Mapping)
        and termination.get("reason") == "state_domain_violation"
    )
    if state_domain_failure:
        final_time_s = _plain_scalar(termination.get("t_final_s"))

    outcome = {
        "step": str(step),
        "success": bool(cvode.get("success", False)),
        "status": _plain_scalar(cvode.get("status", "missing_metadata")),
        "message": str(message or "CVODE failed without a message."),
        "nfev": _plain_scalar(cvode.get("nfev")),
        "effective_solver": _plain_scalar(effective_solver),
        "final_time_s": final_time_s,
    }
    if state_domain_failure:
        event = termination.get("event")
        event = event if isinstance(event, Mapping) else {}
        outcome["failure_kind"] = "state_domain_violation"
        outcome["state_domain"] = {
            key: value
            for key, value in (
                ("t_event_s", _plain_scalar(event.get("t_event_s"))),
                ("guard", event.get("metadata")),
                ("diagnostic", event.get("diagnostic")),
            )
            if value is not None
        }
    return outcome


def is_state_domain_failure(solution: Mapping[str, Any] | None) -> bool:
    """Return whether a solve ended on the operational composition guard."""
    if not isinstance(solution, Mapping):
        return False
    termination = solution.get("termination")
    return bool(
        isinstance(termination, Mapping)
        and termination.get("reason") == "state_domain_violation"
    )


def final_solver_failure_summary(
    solution: Mapping[str, Any] | None,
    *,
    step: str,
) -> dict[str, Any] | None:
    """Return failure metadata when the final CVODE attempt was unsuccessful.

    The solution is inspected only after ``Solver.save_results()`` so its final
    time is dimensional and the failed trajectory already exists on disk.
    Missing CVODE outcome metadata fails closed: a freshly solved step cannot
    be considered safe to propagate when its final status is unknown.
    """
    outcome = compact_solver_outcome(solution, step=step)
    if outcome["success"]:
        return None
    outcome.pop("success")
    return outcome


class FinalSolverFailure(RuntimeError):
    """Raised after a failed final CVODE attempt has been saved to disk."""

    def __init__(self, summary: Mapping[str, Any]):
        self.summary = dict(summary)
        final_time = self.summary.get("final_time_s")
        time_text = "unknown time" if final_time is None else f"t={final_time:g} s"
        super().__init__(
            f"Final CVODE solve failed for step {self.summary.get('step')!r} "
            f"(status={self.summary.get('status')!r}, "
            f"solver={self.summary.get('effective_solver')!r}, {time_text}): "
            f"{self.summary.get('message')}"
        )


def require_final_solver_success(
    solution: Mapping[str, Any] | None,
    *,
    step: str,
) -> None:
    """Raise :class:`FinalSolverFailure` for an unsuccessful final attempt."""
    summary = final_solver_failure_summary(solution, step=step)
    if summary is not None:
        raise FinalSolverFailure(summary)


__all__ = [
    "FinalSolverFailure",
    "compact_solver_outcome",
    "final_solver_failure_summary",
    "is_state_domain_failure",
    "require_final_solver_success",
]
