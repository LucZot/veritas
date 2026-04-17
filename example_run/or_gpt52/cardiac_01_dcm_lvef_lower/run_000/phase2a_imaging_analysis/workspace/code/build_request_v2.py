# filename: build_request.py
import json

DATASET = "acdc"

PLAN = {
    "cohort_mode": "groups",
    "groups": ["DCM", "NOR"],
    "structures": ["left heart ventricle"],
    "observations": ["ED", "ES"],
}

OUTPUT_PATH = "segmentation_request.json"

REQUEST_TEMPLATE = {
    "identifiers": [],
    "structures": PLAN["structures"],
    "results_database": "$BIO_DATA_ROOT/sat_cache/acdc",
    "modality": "mri",
    "model_variant": "nano",
    "chunk_size": 64,
}

def main():
    # Query patients per group
    group_to_patients = {}
    for group in PLAN["groups"]:
        res = list_dataset_patients(DATASET, group=group)
        patients = res.get("patients", []) if isinstance(res, dict) else []
        if not patients:
            raise RuntimeError(f"No patients returned for dataset={DATASET}, group={group}")

        patient_ids = []
        for p in patients:
            pid = p.get("patient_id")
            if not pid:
                continue
            pid = str(pid).strip()
            if ":" in pid:
                raise RuntimeError(f"Invalid patient_id contains ':': {pid}")
            patient_ids.append(pid)

        patient_ids = sorted(set(patient_ids))
        if not patient_ids:
            raise RuntimeError(f"No valid patient_id values found for dataset={DATASET}, group={group}")
        group_to_patients[group] = patient_ids

    identifiers = []
    excluded_patients = []  # {"group":..., "patient_id":..., "reason":...}
    warnings = []           # non-fatal notes

    # Build identifiers ONLY for patients that have BOTH ED and ES available in metadata
    required_obs = list(PLAN["observations"])
    for group in PLAN["groups"]:
        for patient_id in group_to_patients[group]:
            meta = get_patient_metadata(DATASET, patient_id) or {}
            meta_identifiers = meta.get("identifiers", {}) or {}

            # Optional consistency check
            meta_group = meta.get("group", None)
            if meta_group is not None and meta_group != group:
                warnings.append(
                    {"group": group, "patient_id": patient_id, "warning": f"metadata group mismatch (meta={meta_group})"}
                )

            missing_obs = [obs for obs in required_obs if not meta_identifiers.get(obs)]
            if missing_obs:
                excluded_patients.append(
                    {"group": group, "patient_id": patient_id, "reason": f"missing observations: {missing_obs}"}
                )
                continue

            # Include both ED and ES identifiers (paired)
            for obs in required_obs:
                identifiers.append(f"{DATASET}:{patient_id}:{obs}")

    # Must not be empty, and each group must contribute at least one included patient
    included_patient_counts = {g: 0 for g in PLAN["groups"]}
    for entry in excluded_patients:
        pass  # just to keep structure clear

    # derive included patients by group
    for group in PLAN["groups"]:
        excluded_set = {e["patient_id"] for e in excluded_patients if e["group"] == group}
        included = [pid for pid in group_to_patients[group] if pid not in excluded_set]
        included_patient_counts[group] = len(included)

    if not identifiers:
        raise RuntimeError(
            "No identifiers generated after enforcing ED+ES pairing. "
            "Check segmentation_request_build_log.json for exclusion reasons."
        )

    for g, n_included in included_patient_counts.items():
        if n_included == 0:
            raise RuntimeError(f"Group {g} has 0 included patients after ED+ES pairing; refusing to write request.")

    # Stable ordering (group -> patient -> observation), and de-duplicate defensively
    identifiers = list(dict.fromkeys(identifiers))

    req = dict(REQUEST_TEMPLATE)
    req["identifiers"] = identifiers

    with open(OUTPUT_PATH, "w") as f:
        json.dump(req, f, indent=2)

    # Build log for QC/traceability
    group_patient_counts = {g: len(pids) for g, pids in group_to_patients.items()}
    log = {
        "dataset": DATASET,
        "plan_groups": PLAN["groups"],
        "plan_observations": PLAN["observations"],
        "n_identifiers": len(identifiers),
        "group_patient_counts": group_patient_counts,
        "group_included_patient_counts": included_patient_counts,
        "n_excluded_patients": len(excluded_patients),
        "excluded_patients_preview": excluded_patients[:50],
        "n_warnings": len(warnings),
        "warnings_preview": warnings[:50],
        "note": "Patients are included only if ALL required observations are present (ED+ES pairing enforced).",
    }
    with open("segmentation_request_build_log.json", "w") as f:
        json.dump(log, f, indent=2)

if __name__ == "__main__":
    main()