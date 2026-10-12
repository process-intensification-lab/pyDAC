"""Canonical TVSA step names, aliases, and option normalization."""

from __future__ import annotations

from pydac.utils.tvsa.termination import normalize_termination_mode


STEP_LABEL_BY_KEY = {
    "startup": "Startup",
    "blow": "Blowdown",
    "heat": "Wall pre-heating",
    "w_purge": "Wall purge",
    "s_purge": "Steam purge",
    "w_des": "Desorption (wall-heated)",
    "s_des": "Desorption (steam-assisted)",
    "press": "Pressurization",
    "cool": "Cooldown",
    "ads": "Adsorption",
}

STEP_UNIT_BY_KEY = {
    "startup": "hr",
    "blow": "min",
    "heat": "hr",
    "w_purge": "hr",
    "w_des": "hr",
    "s_des": "hr",
    "press": "min",
    "cool": "min",
    "s_purge": "min",
    "ads": "hr",
}

STEP_ALIAS_BY_TOKEN = {
    "startup": "startup",
    "start": "startup",
    "initialization": "startup",
    "ads": "ads",
    "adsorption": "ads",
    "blow": "blow",
    "blowdown": "blow",
    "heat": "heat",
    "preheat": "heat",
    "pre_heating": "heat",
    "preheating": "heat",
    "w_purge": "w_purge",
    "wall_purge": "w_purge",
    "wall_heated_purge": "w_purge",
    "wall_purging": "w_purge",
    "purge": "s_purge",
    "s_purge": "s_purge",
    "steam_purge": "s_purge",
    "pre_purge": "s_purge",
    "steam_purging": "s_purge",
    "w_des": "w_des",
    "wall_heated_desorption": "w_des",
    "wall_desorption": "w_des",
    "s_des": "s_des",
    "steam_assisted_desorption": "s_des",
    "steam_desorption": "s_des",
    "cool": "cool",
    "cooldown": "cool",
    "cooling": "cool",
    "press": "press",
    "pressurization": "press",
    "pressurisation": "press",
}

CANONICAL_STEP_KEYS = frozenset(STEP_ALIAS_BY_TOKEN.values())
SUPPORTED_STEP_TOKENS = frozenset(STEP_ALIAS_BY_TOKEN)
GENERIC_DESORPTION_STEP_TOKENS = frozenset({"des", "desorption"})
DESORPTION_STEP_ALIASES = {
    token: step
    for token, step in STEP_ALIAS_BY_TOKEN.items()
    if step in {"w_des", "s_des"}
}


def normalize_step_token_text(step_token):
    """Normalize case and separators without validating the resulting token."""
    return str(step_token).strip().lower().replace("-", "_").replace(" ", "_")


def normalize_step_token(token):
    """Normalize a user-provided token to a canonical TVSA step key."""
    norm = normalize_step_token_text(token)

    if norm in GENERIC_DESORPTION_STEP_TOKENS:
        raise ValueError(
            f"Generic desorption token '{token}' is not allowed. "
            "Use an explicit desorption step: `w_des`/`wall_heated_desorption` "
            "or `s_des`/`steam_assisted_desorption`."
        )

    step = STEP_ALIAS_BY_TOKEN.get(norm, norm)
    if step not in CANONICAL_STEP_KEYS:
        valid_tokens = sorted(STEP_ALIAS_BY_TOKEN)
        raise ValueError(
            f"Unknown cycle step token '{token}'. "
            f"Valid tokens: {', '.join(valid_tokens)}."
        )
    return step


def normalize_desorption_step(step_token):
    """Normalize an explicit wall-heated or steam-assisted desorption token."""
    if step_token is None:
        return None

    token = normalize_step_token_text(step_token)
    if token in GENERIC_DESORPTION_STEP_TOKENS:
        raise ValueError(
            f"Generic desorption token '{step_token}' is not allowed. "
            "Use an explicit desorption step: `w_des`/`wall_heated_desorption` "
            "or `s_des`/`steam_assisted_desorption`."
        )

    canonical = DESORPTION_STEP_ALIASES.get(token)
    if canonical is None:
        valid = ", ".join(sorted(DESORPTION_STEP_ALIASES))
        raise ValueError(
            f"Unknown desorption step '{step_token}'. "
            f"Valid explicit desorption tokens: {valid}."
        )
    return canonical


def normalize_step_options(step_options):
    """Normalize per-step option keys and termination-mode values."""
    if step_options is None:
        return {}

    normalized = {}
    for step_token, opts in dict(step_options).items():
        step_key = normalize_step_token(step_token)
        existing = normalized.get(step_key, {})
        merged = dict(existing)
        opts = dict(opts)
        if "termination_mode" in opts:
            opts["termination_mode"] = normalize_termination_mode(
                opts["termination_mode"],
                option_name=f"step_options['{step_token}']['termination_mode']",
                allow_none=False,
            )
        merged.update(opts)
        normalized[step_key] = merged

    return normalized


__all__ = [
    "CANONICAL_STEP_KEYS",
    "DESORPTION_STEP_ALIASES",
    "GENERIC_DESORPTION_STEP_TOKENS",
    "STEP_ALIAS_BY_TOKEN",
    "STEP_LABEL_BY_KEY",
    "STEP_UNIT_BY_KEY",
    "SUPPORTED_STEP_TOKENS",
    "normalize_desorption_step",
    "normalize_step_options",
    "normalize_step_token",
    "normalize_step_token_text",
]
