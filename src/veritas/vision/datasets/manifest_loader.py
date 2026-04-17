"""Manifest-driven dataset loader for simple, dataset-agnostic metadata access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ManifestDatasetLoader:
    """Loader that reads a dataset manifest with samples and observations."""

    _manifest_filenames = (
        "dataset_manifest.json",
        "dataset_index.json",
        "dataset.json",
    )

    def __init__(
        self,
        data_root: str | Path,
        dataset_name: Optional[str] = None,
        manifest_path: Optional[str | Path] = None,
    ):
        self.data_root = Path(data_root).expanduser()
        self.manifest_path = self._resolve_manifest_path(manifest_path)
        self.manifest = self._load_manifest()
        self.name = self._resolve_name(dataset_name)
        self.samples = self._normalize_samples()
        self._sample_index = {sample["sample_id"]: sample for sample in self.samples}

    def _resolve_manifest_path(self, manifest_path: Optional[str | Path]) -> Path:
        if manifest_path:
            path = Path(manifest_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Dataset manifest not found: {path}")
            return path

        candidate_roots = [self.data_root]
        if self.data_root.name == "database":
            candidate_roots.append(self.data_root.parent)
        if (self.data_root / "database").is_dir():
            candidate_roots.append(self.data_root / "database")

        for root in candidate_roots:
            for filename in self._manifest_filenames:
                candidate = root / filename
                if candidate.exists():
                    return candidate

        searched = ", ".join(
            str(root / filename)
            for root in candidate_roots
            for filename in self._manifest_filenames
        )
        raise FileNotFoundError(
            "Dataset manifest not found. Looked for: " + searched
        )

    def _load_manifest(self) -> Dict[str, Any]:
        with open(self.manifest_path, "r") as f:
            return json.load(f)

    def _resolve_name(self, dataset_name: Optional[str]) -> str:
        manifest_name = self.manifest.get("name")
        if dataset_name and manifest_name:
            if dataset_name.lower() != str(manifest_name).lower():
                raise ValueError(
                    f"Dataset name '{dataset_name}' does not match manifest name '{manifest_name}'."
                )
        return str(manifest_name or dataset_name or "dataset")

    def _normalize_samples(self) -> List[Dict[str, Any]]:
        samples_raw = (
            self.manifest.get("samples")
            or self.manifest.get("patients")
            or self.manifest.get("subjects")
            or []
        )
        if not isinstance(samples_raw, list):
            raise ValueError("Dataset manifest samples must be a list.")

        normalized = []
        for entry in samples_raw:
            if not isinstance(entry, dict):
                raise ValueError("Dataset manifest sample entries must be objects.")
            sample_id = (
                entry.get("sample_id")
                or entry.get("patient_id")
                or entry.get("id")
            )
            if not sample_id:
                raise ValueError("Dataset manifest sample missing an id.")

            group = (
                entry.get("group")
                or entry.get("cohort")
                or entry.get("pathology")
                or entry.get("label")
            )

            observations_raw = (
                entry.get("observations")
                or entry.get("assets")
                or entry.get("images")
            )
            observations = self._normalize_observations(observations_raw, sample_id)

            reserved = {
                "sample_id",
                "patient_id",
                "id",
                "group",
                "cohort",
                "pathology",
                "label",
                "observations",
                "assets",
                "images",
                "metadata",
            }
            metadata = dict(entry.get("metadata") or {})
            for key, value in entry.items():
                if key not in reserved:
                    metadata[key] = value

            normalized.append(
                {
                    "sample_id": str(sample_id),
                    "group": group,
                    "observations": observations,
                    "metadata": metadata,
                }
            )
        return normalized

    def _normalize_observations(
        self,
        observations_raw: Any,
        sample_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        if observations_raw is None:
            return {}
        if isinstance(observations_raw, dict):
            observations = {}
            for obs_id, value in observations_raw.items():
                path, metadata = self._extract_observation(value, sample_id, obs_id)
                observations[str(obs_id)] = {
                    "path": path,
                    "metadata": metadata,
                }
            return observations
        if isinstance(observations_raw, list):
            observations = {}
            for value in observations_raw:
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Observation entries for '{sample_id}' must be objects."
                    )
                obs_id = (
                    value.get("id")
                    or value.get("observation_id")
                    or value.get("name")
                )
                if not obs_id:
                    raise ValueError(
                        f"Observation entry for '{sample_id}' missing id."
                    )
                path, metadata = self._extract_observation(value, sample_id, obs_id)
                observations[str(obs_id)] = {
                    "path": path,
                    "metadata": metadata,
                }
            return observations
        raise ValueError(
            f"Observations for '{sample_id}' must be a dict or list."
        )

    def _extract_observation(
        self,
        value: Any,
        sample_id: str,
        obs_id: str,
    ) -> tuple[str, Dict[str, Any]]:
        if isinstance(value, dict):
            path = (
                value.get("path")
                or value.get("asset_path")
                or value.get("file")
            )
            if not path:
                raise ValueError(
                    f"Observation '{obs_id}' for '{sample_id}' missing path."
                )
            metadata = {
                key: val
                for key, val in value.items()
                if key not in {"path", "asset_path", "file"}
            }
            return str(path), metadata

        if isinstance(value, str):
            return value, {}

        raise ValueError(
            f"Observation '{obs_id}' for '{sample_id}' has invalid format."
        )

    def list_patients_by_group(
        self,
        group: Optional[str] = None,
        metadata_filters: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for sample in self.samples:
            sample_group = sample.get("group")
            if group is not None and sample_group != group:
                continue
            # Apply metadata filters
            if metadata_filters:
                meta = sample.get("metadata") or {}
                if not all(str(meta.get(k)) == str(v) for k, v in metadata_filters.items()):
                    continue
            results.append(
                {
                    "patient_id": sample["sample_id"],
                    "group": sample_group,
                    "num_frames": len(sample.get("observations") or {}),
                }
            )
        return results

    def get_patient_metadata(self, patient_id: str) -> Dict[str, Any]:
        sample = self._sample_index.get(patient_id)
        if not sample:
            raise FileNotFoundError(
                f"Patient '{patient_id}' not found in dataset manifest."
            )
        observations = sample.get("observations") or {}
        identifiers = {
            obs_id: f"{self.name}:{patient_id}:{obs_id}"
            for obs_id in observations.keys()
        }
        result = {
            "patient_id": patient_id,
            "group": sample.get("group"),
            "num_frames": len(observations),
            "observations": list(observations.keys()),
            "identifiers": identifiers,
        }
        # Flatten metadata fields to top level for easy access
        # (e.g., meta["idh_status"] instead of meta["metadata"]["idh_status"])
        for key, value in (sample.get("metadata") or {}).items():
            if key not in result:
                result[key] = value
        return result

    def resolve_identifier(self, identifier: str) -> str:
        parts = identifier.split(":")
        if len(parts) != 3:
            return identifier
        dataset_name, patient_id, observation_id = parts
        if dataset_name.lower() != self.name.lower():
            raise ValueError(
                f"Identifier dataset '{dataset_name}' does not match '{self.name}'."
            )
        sample = self._sample_index.get(patient_id)
        if not sample:
            raise FileNotFoundError(
                f"Patient '{patient_id}' not found in dataset manifest."
            )
        observations = sample.get("observations") or {}
        observation = observations.get(observation_id)
        if not observation:
            raise FileNotFoundError(
                f"Observation '{observation_id}' not found for '{patient_id}'."
            )
        path_value = observation.get("path")
        if not path_value:
            raise FileNotFoundError(
                f"Observation '{observation_id}' for '{patient_id}' has no path."
            )
        resolved = Path(path_value)
        if not resolved.is_absolute():
            resolved = self.manifest_path.parent / resolved
        return str(resolved)
