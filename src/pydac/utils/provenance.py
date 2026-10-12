"""Portable execution provenance activity records, independent of numerical artifact schemas.

Local Git is only a best-effort dirty-state observation. It is never evidence
that a revision is publicly retrievable, so runtime HEADs are not exported.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import re
import subprocess
import sys
import tempfile
from uuid import UUID, uuid4

from pydac.config.errors import ConfigurationError, RunRefused
from pydac.utils.solver.flags import env_flag
from pydac.utils.version import __version__, PUBLIC_SOURCE

SCHEMA_VERSION = 1
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ACTIVITIES = {"solve", "step_metrics", "cycle_report", "optimization", "sweep", "analysis", "run"}
_KEYS = {"schema_version", "record_id", "created_at", "activity", "software",
         "input", "execution", "sources", "artifacts"}
_DEPENDENCIES = {"numpy": "numpy", "scipy": "scipy", "scikit-sundae": "sksundae",
                 "numba": "numba", "pymoo": "pymoo", "pandas": "pandas"}
_SOLVER_SWITCHES = (
    ("PYDAC_RHS_CSR_OPERATORS", "pydac.core.column_model", "_RHS_CSR_OPERATORS"),
    ("PYDAC_CACHE_SLICE_WRITE", "pydac.core.pymrm_caching", "_CACHE_SLICE_WRITE"),
)
_SOFTWARE_KEYS = {"project", "version", "version_source", "model_generation", "python",
                  "platform", "dependencies", "threads", "native_libraries",
                  "native_build_identity", "solver_environment", "git", "public_source"}


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def checkout_root(package=PACKAGE_ROOT):
    """Return the source tree holding ``package`` (``<root>/src/pydac``), else None.

    An installed wheel lives in site-packages and has no such tree.
    """
    package = Path(package)
    root = package.parent.parent
    if package.parent.name != "src" or not (root / "pyproject.toml").is_file() \
            or not (package / "__init__.py").is_file():
        return None
    return root


def git_observation(root=None):
    """Inspect only the source location, with bounded, non-fatal Git calls."""
    root = Path(root) if root is not None else checkout_root()
    result = {"status": "no_checkout", "dirty": None, "revision": None,
              "revision_status": "unavailable"}
    if root is None:
        return result
    # Source archives installed under an unrelated checkout must not inherit it.
    try:
        if not any((p / ".git").exists() for p in (root, root.parent)):
            return result
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env["GIT_OPTIONAL_LOCKS"] = "0"
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", "src/pydac/__init__.py"],
            capture_output=True, timeout=2, check=False, env=env,
        )
        if tracked.returncode:
            return result
        completed = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal", "--", "."],
            capture_output=True, timeout=2, check=False,
            env=env,
        )
        if completed.returncode:
            result["status"] = "inspection_failed"
        else:
            result.update(status="available", dirty=bool(completed.stdout.strip()),
                          revision_status="unverified_or_non_public")
    except FileNotFoundError:
        result["status"] = "git_unavailable"
    except (OSError, subprocess.TimeoutExpired):
        result["status"] = "inspection_failed"
    return result


def software_identity():
    dependencies = {}
    for distribution, module in _DEPENDENCIES.items():
        loaded = getattr(sys.modules.get(module), "__version__", None)
        if loaded is not None:
            version, source = str(loaded), "loaded_module"
        else:
            try:
                version, source = metadata.version(distribution), "distribution_metadata"
            except (metadata.PackageNotFoundError, OSError, UnicodeError):
                version, source = None, "unavailable"
        dependencies[distribution] = {"version": version, "source": source}
    threads = {}
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value = os.environ.get(name)
        threads[name] = value if value is None or value.isdecimal() else "non_numeric"
    for name, choices in (("OMP_WAIT_POLICY", {"ACTIVE", "PASSIVE"}),
                          ("MKL_DYNAMIC", {"TRUE", "FALSE"})):
        value = os.environ.get(name)
        threads[name] = value.upper() if value and value.upper() in choices else None
    native = {"sundials_version": getattr(sys.modules.get("sksundae"), "SUNDIALS_VERSION", None)}
    numpy_config = getattr(getattr(sys.modules.get("numpy"), "__config__", None), "CONFIG", {})
    for name in ("blas", "lapack"):
        dependency = numpy_config.get("Build Dependencies", {}).get(name, {})
        native[name] = {key: dependency.get(key) for key in ("name", "version", "detection method")}
    # The A/B switches that change the arithmetic path; a loaded
    # module's bound value is what the solve used, else the environment.
    solver_environment = {}
    for variable, module, name in _SOLVER_SWITCHES:
        bound = getattr(sys.modules.get(module), name, None)
        solver_environment[variable] = bool(bound) if bound is not None else env_flag(variable)
    return {
        "project": "pyDAC", "version": __version__, "version_source": "package_constant",
        "model_generation": 3,
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "dependencies": dependencies, "threads": threads,
        "native_libraries": native, "native_build_identity": None,
        "solver_environment": solver_environment, "git": git_observation(),
        "public_source": deepcopy(PUBLIC_SOURCE),
    }


def case_input(config=None):
    if config is None:
        return {"status": "unknown", "resolved_config": None, "case_metadata": None,
                "source_case_metadata": None, "relocated_fields": []}
    from pydac.inputs.case_io import case_metadata, validate_case_document
    document = validate_case_document(config)
    original = case_metadata(document)
    # Output placement is not a physical input. Record both fingerprints rather
    # than claiming the relocatable copy is byte-identical to the source case.
    document["execution"]["output_root"] = "."
    return {"status": "recorded", "resolved_config": document,
            "case_metadata": case_metadata(document), "source_case_metadata": original,
            "relocated_fields": ["execution.output_root"]}


def new_record(activity, *, config=None, execution=None, sources=None, artifacts=None):
    return validate_record({
        "schema_version": SCHEMA_VERSION, "record_id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(), "activity": activity,
        "software": software_identity(), "input": case_input(config),
        "execution": deepcopy(execution or {}), "sources": deepcopy(sources or []),
        "artifacts": deepcopy(artifacts or []),
    })


def validate_record(record):
    """Reject unsupported/malformed present provenance; absence is handled separately."""
    if not isinstance(record, dict) or set(record) != _KEYS:
        raise ValueError("Invalid provenance record fields.")
    if type(record["schema_version"]) is not int or record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported provenance schema version.")
    if record["activity"] not in ACTIVITIES:
        raise ValueError("Invalid provenance activity.")
    try:
        UUID(record["record_id"])
        timestamp = datetime.fromisoformat(record["created_at"])
        if timestamp.utcoffset() is None:
            raise ValueError("Provenance timestamp needs a timezone.")
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid provenance identity/timestamp.") from exc
    for name in ("software", "input", "execution"):
        if not isinstance(record[name], dict):
            raise ValueError(f"Provenance {name} must be a mapping.")
    software = record["software"]
    if (set(software) != _SOFTWARE_KEYS or software.get("project") != "pyDAC"
            or not isinstance(software.get("version"), str) or not software["version"]):
        raise ValueError("Invalid provenance software identity.")
    git = software["git"]
    public_source = software["public_source"]
    if public_source is not None:
        if (not isinstance(public_source, dict) or set(public_source) != {"revision", "url"}
                or not isinstance(public_source["revision"], str)
                or not re.fullmatch(r"[0-9a-f]{40,64}", public_source["revision"])
                or not isinstance(public_source["url"], str)
                or not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/commit/"
                                    + public_source["revision"], public_source["url"])):
            raise ValueError("Invalid qualified public build source.")
    if (not isinstance(git, dict) or set(git) != {"status", "dirty", "revision", "revision_status"}
            or git["status"] not in {"available", "no_checkout", "git_unavailable", "inspection_failed"}
            or git["revision"] is not None
            or git["revision_status"] not in {"unavailable", "unverified_or_non_public"}
            or (git["dirty"] is not None and type(git["dirty"]) is not bool)):
        raise ValueError("Invalid optional Git provenance.")
    for name in ("python", "platform", "dependencies", "threads", "native_libraries", "solver_environment"):
        if not isinstance(software[name], dict):
            raise ValueError(f"Invalid software {name} record.")
    inputs = record["input"]
    if set(inputs) != {"status", "resolved_config", "case_metadata", "source_case_metadata", "relocated_fields"}:
        raise ValueError("Invalid provenance input fields.")
    if inputs.get("status") == "recorded":
        # Hash the case as stored, not as today's vocabulary would validate it,
        # so runs stay readable when a later version adds a case field.
        from pydac.inputs.case_io import stored_case_metadata
        if stored_case_metadata(inputs.get("resolved_config")) != inputs.get("case_metadata"):
            raise ValueError("Provenance case fingerprint mismatch.")
    elif inputs.get("status") != "unknown" or inputs.get("resolved_config") is not None:
        raise ValueError("Invalid provenance input status.")
    for name in ("sources", "artifacts"):
        if not isinstance(record[name], list):
            raise ValueError(f"Provenance {name} must be a list.")
    for source in record["sources"]:
        if not isinstance(source, dict):
            raise ValueError("Invalid provenance source.")
        if source.get("record") is not None:
            parent = validate_record(source["record"])
            if source.get("record_sha256") != hashlib.sha256(_json_bytes(parent)).hexdigest():
                raise ValueError("Provenance source record hash mismatch.")
        if "artifact" in source:
            _validate_artifact(source["artifact"])
    for artifact in record["artifacts"]:
        _validate_artifact(artifact)
    _json_bytes(record)
    return deepcopy(record)


def _validate_artifact(artifact):
    name = artifact.get("name") if isinstance(artifact, dict) else None
    if (not isinstance(name, str) or not name or Path(name).is_absolute()
            or PureWindowsPath(name).is_absolute() or ".." in PureWindowsPath(name).parts):
        raise ValueError("Provenance artifact names must be relative.")
    if not isinstance(artifact.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]):
        raise ValueError("Invalid provenance artifact digest.")


def source_record(record=None):
    if record is None:
        return {"status": "unknown", "record": None, "record_sha256": None}
    record = validate_record(record)
    return {"status": "recorded", "record": record,
            "record_sha256": hashlib.sha256(_json_bytes(record)).hexdigest()}


def atomic_json(path, document):
    path = Path(path)
    payload = _json_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".provenance-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def save_record(folder, record, *, pointer=None):
    record = validate_record(record)
    folder = Path(folder)
    path = folder / "provenance" / f"{record['record_id']}.json"
    if path.exists() and path.read_bytes() != _json_bytes(record):
        raise ValueError("Refusing to replace an existing provenance identity.")
    atomic_json(path, record)
    if pointer is not None:
        atomic_json(folder / pointer, record)
    return path


def read_record(path):
    path = Path(path)
    if not path.exists():
        return None
    return validate_record(json.loads(path.read_text(encoding="utf-8")))


def file_artifact(path, *, name=None):
    path = Path(path)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"name": name or path.name, "sha256": digest}


def metrics_sources(folder, steps):
    """Snapshot loaded KPI artifact identity, validating a present sidecar binding."""
    sources = []
    for step in steps:
        path = Path(folder) / step / "kpis.pkl"
        artifact = file_artifact(path, name=f"{step}/kpis.pkl")
        record = read_record(path.with_name("metrics_provenance.json"))
        if record is not None:
            expected = next((a for a in record["artifacts"] if a["name"] == "kpis.pkl"), None)
            if expected is None or expected["sha256"] != artifact["sha256"]:
                raise ValueError("KPI artifact no longer matches its provenance.")
        sources.append({**source_record(record), "artifact": artifact})
    return sources


def recorded_versions(folder, activity):
    """pyDAC versions of the ``activity`` records saved in ``folder/provenance``.

    Only the identity fields are read, so records of other versions (or a
    later execution provenance schema) still answer which version wrote them.
    """
    versions = set()
    for path in sorted((Path(folder) / "provenance").glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["activity"] == activity:
                version = record["software"]["version"]
                if not isinstance(version, str) or not version:
                    raise TypeError("missing version")
                versions.add(version)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ConfigurationError(f"Unreadable provenance record {path}: {exc}") from exc
    return versions


def require_same_version(folder, activity, what, *, still_allowed="Reporting it"):
    """Refuse to continue a run that another pyDAC version wrote.

    Runs saved before provenance recorded the software carry no record and
    are not refused.
    """
    others = sorted(recorded_versions(folder, activity) - {__version__})
    if others:
        raise RunRefused(
            f"{what} was written by pyDAC {', '.join(others)}; this is pyDAC {__version__}. "
            f"{still_allowed} still works; install that version to continue it, or start a new run."
        )


def start_activity(folder, activity, *, config=None, execution=None, sources=None):
    record = new_record(activity, config=config, execution=execution, sources=sources)
    save_record(folder, record)
    return record
