# filename: build_request.py
import json

# Configuration from the plan
DATASET = "acdc"
GROUPS = ["DCM", "NOR"]
STRUCTURES = ["left heart ventricle"]
OBSERVATIONS = ["ED", "ES"]
RESULTS_DATABASE = "$BIO_DATA_ROOT/sat_cache/acdc"
MODALITY = "mri"
MODEL_VARIANT = "nano"
CHUNK_SIZE = 64

def main():
    identifiers = []

    # Iterate over each group specified in the plan
    for group in GROUPS:
        # Query patients for the current group
        patients_result = list_dataset_patients(DATASET, group=group)
        patients = patients_result.get("patients", [])

        # Process each patient
        for patient in patients:
            patient_id = patient.get("patient_id")
            if not patient_id:
                continue  # Skip if no patient_id

            # Retrieve metadata for the patient
            meta = get_patient_metadata(DATASET, patient_id)

            # For each observation, construct the identifier string
            for obs in OBSERVATIONS:
                # The identifiers dict in metadata maps observation names to IDs
                obs_id = meta.get("identifiers", {}).get(obs)
                if not obs_id:
                    # If the observation identifier is missing, skip this entry
                    continue
                identifier_str = f"{DATASET}:{patient_id}:{obs}"
                identifiers.append(identifier_str)

    # Build the final request dictionary
    request = {
        "identifiers": identifiers,
        "structures": STRUCTURES,
        "results_database": RESULTS_DATABASE,
        "modality": MODALITY,
        "model_variant": MODEL_VARIANT,
        "chunk_size": CHUNK_SIZE
    }

    # Write the request to segmentation_request.json
    with open("segmentation_request.json", "w") as f:
        json.dump(request, f, indent=2)

if __name__ == "__main__":
    main()