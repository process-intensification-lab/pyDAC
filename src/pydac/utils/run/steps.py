"""Saved-result inference for canonical TVSA steps."""

import os

from pydac.utils.tvsa.steps import (
    CANONICAL_STEP_KEYS,
    DESORPTION_STEP_ALIASES,
    GENERIC_DESORPTION_STEP_TOKENS,
    normalize_desorption_step,
    normalize_step_token,
    normalize_step_token_text,
)


def infer_desorption_step(cycle_steps=None, default="s_des"):
    # `s_des` matches the vanilla cycle in `utils/tvsa/cycle_steps.py`, so a
    # caller that passes no explicit sequence infers the step that sequence runs.
    default_step = normalize_desorption_step(default) if default is not None else None
    if cycle_steps is None:
        return default_step

    found = []
    for step in cycle_steps:
        token = normalize_step_token_text(step)
        if token in GENERIC_DESORPTION_STEP_TOKENS:
            raise ValueError(
                f"Generic desorption token '{step}' is not allowed. "
                "Use an explicit desorption step: `w_des`/`wall_heated_desorption` "
                "or `s_des`/`steam_assisted_desorption`."
            )

        canonical = DESORPTION_STEP_ALIASES.get(token)
        if canonical is not None and canonical not in found:
            found.append(canonical)

    if len(found) > 1:
        raise ValueError(
            "cycle_steps contains both steam-assisted and wall-heated desorption tokens. "
            "Choose one explicit desorption mode."
        )
    if found:
        return found[0]
    return default_step


def is_steam_assisted_desorption_step(desorption_step):
    return normalize_desorption_step(desorption_step) == "s_des"


def infer_results_base_folder(base_folder):
    entries = {
        name for name in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, name))
    }
    found = [name for name in ("2D_mode", "1D_mode") if name in entries]
    if len(found) > 1:
        raise ValueError(
            f"Multiple spatial-mode result folders found in {base_folder}: {found}. "
            "Expected exactly one active results folder."
        )
    if found:
        return os.path.join(base_folder, found[0])
    return base_folder


def infer_desorption_step_from_results(base_folder):
    results_base_folder = infer_results_base_folder(base_folder)
    entries = {
        name for name in os.listdir(results_base_folder)
        if os.path.isdir(os.path.join(results_base_folder, name))
    }
    found = [step for step in ("w_des", "s_des") if step in entries]
    if len(found) > 1:
        raise ValueError(
            f"Both desorption variants found in {results_base_folder}: {found}. "
            "Expected exactly one explicit desorption step in the saved results."
        )
    if found:
        return found[0]

    cycle_dirs = []
    for name in entries:
        if not name.startswith("cycle_"):
            continue
        suffix = name.split("_", 1)[1]
        if suffix.isdigit():
            cycle_dirs.append((int(suffix), name))

    if cycle_dirs:
        _, last_cycle = max(cycle_dirs)
        return infer_desorption_step_from_results(os.path.join(results_base_folder, last_cycle))

    if results_base_folder != base_folder:
        root_entries = {
            name for name in os.listdir(base_folder)
            if os.path.isdir(os.path.join(base_folder, name))
        }
        root_cycle_dirs = []
        for name in root_entries:
            if not name.startswith("cycle_"):
                continue
            suffix = name.split("_", 1)[1]
            if suffix.isdigit():
                root_cycle_dirs.append((int(suffix), name))

        if root_cycle_dirs:
            _, last_cycle = max(root_cycle_dirs)
            return infer_desorption_step_from_results(os.path.join(base_folder, last_cycle))

    return None


def ordered_saved_step_folders(base_folder, preferred=()):
    """Yield saved folder names with canonical step keys in preferred order."""
    rank = {step: index for index, step in enumerate(dict.fromkeys(preferred))}
    candidates = []
    for folder_name in os.listdir(base_folder):
        if not os.path.isdir(os.path.join(base_folder, folder_name)):
            continue
        if folder_name not in CANONICAL_STEP_KEYS:
            continue
        candidates.append((rank.get(folder_name, len(rank)), folder_name))
    for _, folder_name in sorted(candidates):
        yield folder_name, folder_name
