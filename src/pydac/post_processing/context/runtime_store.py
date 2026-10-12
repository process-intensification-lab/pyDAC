"""Persistence adapter for the validated human-readable runtime report."""

from __future__ import annotations

import json
from pathlib import Path

from pydac.utils.postprocessing.runtime_report import (
    RUNTIME_REPORT_FILENAME,
    merge_runtime_reports,
    runtime_report_from_model,
    validate_runtime_report,
)


class RuntimeReportStore:
    """Save one deterministic report whose completed step blocks are immutable."""

    def __init__(self, base_folder):
        self.base_folder = Path(base_folder)

    @property
    def path(self):
        return self.base_folder / RUNTIME_REPORT_FILENAME

    def exists(self):
        return self.path.is_file()

    def from_model(self, model):
        return runtime_report_from_model(model)

    def validate_compatible(self, report):
        incoming = validate_runtime_report(report)
        if not self.exists():
            return incoming
        return merge_runtime_reports(self.load(), incoming)

    def save(self, report):
        merged = self.validate_compatible(report)
        self.base_folder.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(merged, indent=2, sort_keys=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Resolved runtime report successfully saved to '{self.path}'")
        return self.path

    def load(self):
        if not self.exists():
            raise FileNotFoundError(f"No resolved runtime report found at '{self.path}'.")
        with self.path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        return validate_runtime_report(document)


__all__ = ["RuntimeReportStore"]
