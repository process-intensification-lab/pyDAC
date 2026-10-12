"""Persistence adapter for typed per-step metrics."""

from pathlib import Path
import pickle

import pandas as pd

from pydac.post_processing.metrics.step_metrics import StepMetrics


class StepMetricsStore:
    """Convert, display, save, and load ``StepMetrics`` without domain logic."""

    def __init__(self, base_folder):
        self.base_folder = Path(base_folder)

    @staticmethod
    def to_dataframe(step_metrics):
        if not isinstance(step_metrics, StepMetrics):
            raise TypeError("step_metrics must be a StepMetrics instance.")
        return step_metrics.to_legacy_dataframe()

    @staticmethod
    def _species_sort_key(species_name):
        species = str(species_name).strip().upper()
        return (1, species) if species == "SYSTEM" else (0, species)

    @staticmethod
    def _metric_sort_key(metric_name):
        metric = str(metric_name).strip().lower()
        explicit_order = {
            "nz [-]": 0,
            "nr [-]": 1,
            "initial mass [kg]": 10,
            "mass change [kg]": 11,
            "state-based mass change [kg]": 12,
            "integrated-state mass gap [kg]": 13,
            "integrated-state error [%]": 14,
            "left-over mass [kg]": 15,
            "state-based left-over mass [kg]": 16,
            "final gas-phase mass [kg]": 17,
            "mass after condenser [kg]": 18,
            "mass after vacuum discharge [kg]": 19,
            "steam mass [kg]": 20,
            "saturation [%]": 22,
        }
        if metric in explicit_order:
            return 0, explicit_order[metric], metric
        if "[kg]" in metric:
            return 1, 100, metric
        if "[kg/s]" in metric:
            return 2, 100, metric
        if "[mj" in metric:
            return 3, 100, metric
        return 4, 100, metric

    @classmethod
    def sort_dataframe(cls, dataframe):
        if dataframe.empty:
            return dataframe.copy()
        table = dataframe.copy()
        declared_species = tuple(
            dict.fromkeys(
                str(value).strip().upper()
                for value in table["Species"]
                if str(value).strip().upper() != "SYSTEM"
            )
        )
        rank = {species: index for index, species in enumerate(declared_species)}
        table["_species_rank"] = table["Species"].map(
            lambda value: rank.get(str(value).strip().upper(), len(rank))
        )
        table["_metric_rank"] = table["KPI"].map(cls._metric_sort_key)
        return (
            table.sort_values(by=["Step", "_species_rank", "_metric_rank"])
            .drop(columns=["_species_rank", "_metric_rank"])
            .reset_index(drop=True)
        )

    def show(self, step_metrics):
        dataframe = self.sort_dataframe(self.to_dataframe(step_metrics))
        print("\n" + "-" * 60)
        title = f" Step Metrics - {step_metrics.step.capitalize()} Step"
        print(title.center(60))
        print("-" * 60)
        print(dataframe.to_string(index=False))

    def save(self, step_metrics, step_folder=None, *, source_provenance=None):
        from pydac.utils.provenance import file_artifact, new_record, save_record, source_record
        source = source_record(source_provenance)
        dataframe = self.to_dataframe(step_metrics)
        folder_name = step_metrics.step if step_folder is None else str(step_folder)
        save_dir = self.base_folder / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "kpis.pkl"
        csv_path = save_dir / "kpis.csv"

        with save_path.open("wb") as handle:
            pickle.dump(dataframe, handle)
        dataframe.to_csv(csv_path, index=False)

        provenance = new_record(
            "step_metrics", sources=[source],
            execution={"step": folder_name},
            artifacts=[file_artifact(save_path), file_artifact(csv_path)],
        )
        save_record(save_dir, provenance, pointer="metrics_provenance.json")

        print(f"Step metrics successfully saved to '{save_path}' and '{csv_path}'")
        return save_path

    def load(self, step_folder):
        load_path = self.base_folder / str(step_folder) / "kpis.pkl"
        if not load_path.is_file():
            raise FileNotFoundError(f"No saved step metrics found at '{load_path}'.")

        with load_path.open("rb") as handle:
            dataframe = pickle.load(handle)
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame in '{load_path}'.")
        return StepMetrics.from_legacy_dataframe(dataframe)
