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
        # normalize to patient_id list
        patient_ids = [p["patient_id"] for p in patients if "patient_id" in p]
        patient_ids = sorted(set(patient_ids))
        if not patient_ids:
            raise RuntimeError(f"No patient_id values found for dataset={DATASET}, group={group}")
        group_to_patients[group] = patient_ids

    identifiers = []
    missing = []  # (group, patient_id, obs, reason)

    # Build identifiers in required format: acdc:<patient_id>:<observation>
    for group in PLAN["groups"]:
        for patient_id in group_to_patients[group]:
            meta = get_patient_metadata(DATASET, patient_id) or {}
            meta_group = meta.get("group", None)
            if meta_group is not None and meta_group != group:
                # Not fatal, but record for traceability
                missing.append((group, patient_id, None, f"metadata group mismatch (meta={meta_group})"))

            meta_identifiers = meta.get("identifiers", {}) or {}
            for obs in PLAN["observations"]:
                # Ensure the observation exists for this patient (if metadata provides that info)
                # We still construct the request identifier in the mandated format.
                if obs not in meta_identifiers or meta_identifiers.get(obs) in (None, ""):
                    missing.append((group, patient_id, obs, "missing observation identifier in metadata"))
                    continue

                identifiers.append(f"{DATASET}:{patient_id}:{obs}")

    # Sanity checks: ensure we didn't end up empty, and ensure each group contributed something
    if not identifiers:
        details = "\n".join([str(x) for x in missing[:50]])
        raise RuntimeError(
            "No identifiers generated. First missing entries (up to 50):\n" + details
        )

    # Ensure both groups have at least one identifier
    group_contrib = {g: 0 for g in PLAN["groups"]}
    for ident in identifiers:
        # ident format: acdc:patientXXX:ED
        _, pid, _ = ident.split(":")
        # determine group by lookup
        for g, pids in group_to_patients.items():
            if pid in pids:
                group_contrib[g] += 1
                break

    for g, cnt in group_contrib.items():
        if cnt == 0:
            raise RuntimeError(f"Group {g} contributed 0 identifiers; refusing to write request.")

    # De-duplicate and keep stable ordering
    identifiers = sorted(set(identifiers), key=lambda s: (s.split(":")[1], s.split(":")[2]))

    req = dict(REQUEST_TEMPLATE)
    req["identifiers"] = identifiers

    with open(OUTPUT_PATH, "w") as f:
        json.dump(req, f, indent=2)

    # Optional: write a small log for debugging in the sandbox
    log = {
        "dataset": DATASET,
        "plan_groups": PLAN["groups"],
        "plan_observations": PLAN["observations"],
        "n_identifiers": len(identifiers),
        "group_patient_counts": {g: len(pids) for g, pids in group_to_patients.items()},
        "group_identifier_counts": group_contrib,
        "n_missing": len(missing),
        "missing_preview": missing[:25],
    }
    with open("segmentation_request_build_log.json", "w") as f:
        json.dump(log, f, indent=2)

if __name__ == "__main__":
    main()