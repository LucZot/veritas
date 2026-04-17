import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind
import os
from pathlib import Path

# Set up paths
results_db_path = "$BIO_DATA_ROOT/sat_cache/acdc"
output_dir = Path("data")
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

# Load all patient IDs
patient_ids = sat.list_patients(results_db_path)

# Filter patients by groups DCM and NOR
filtered_patients = []
for patient_id in patient_ids:
    metadata = sat.get_patient_metadata(patient_id)
    if metadata['group'] in ['DCM', 'NOR']:
        filtered_patients.append(patient_id)

# Initialize lists to store results
dcms = []
nors = []

# Process each patient
for patient_id in filtered_patients:
    metadata = sat.get_patient_metadata(patient_id)
    group = metadata['group']

    # Get observation identifiers
    obs_map = sat.get_observation_identifiers(patient_id)

    # Load masks for ED and ES
    try:
        ed_masks = sat.load_structure_mask(results_db_path, patient_id, 'left heart ventricle', source_image_contains=obs_map['ED'])
        es_masks = sat.load_structure_mask(results_db_path, patient_id, 'left heart ventricle', source_image_contains=obs_map['ES'])

        # Calculate volumes
        if ed_masks and es_masks:
            ed_volume = 0
            es_volume = 0

            # Sum volumes from all masks for ED and ES
            for mask in ed_masks:
                ed_volume += sat.calculate_volume(mask['mask'], mask['spacing'])

            for mask in es_masks:
                es_volume += sat.calculate_volume(mask['mask'], mask['spacing'])

            # Calculate LVEF
            if ed_volume > 0:
                lv_ef = ((ed_volume - es_volume) / ed_volume) * 100
                patient_data = {
                    'patient_id': patient_id,
                    'group': group,
                    'lvef': lv_ef,
                    'ed_volume': ed_volume,
                    'es_volume': es_volume
                }

                if group == 'DCM':
                    dcms.append(patient_data)
                elif group == 'NOR':
                    nors.append(patient_data)
    except Exception as e:
        print(f"Error processing patient {patient_id}: {e}")
        continue

# Prepare data for analysis
dcms_df = pd.DataFrame(dcms)
nors_df = pd.DataFrame(nors)

# Check sample sizes
n_dcm = len(dcms_df)
n_nor = len(nors_df)

print(f"Sample sizes - DCM: {n_dcm}, NOR: {n_nor}")

# Perform two-sample t-test
if n_dcm > 0 and n_nor > 0:
    t_stat, p_value = ttest_ind(dcms_df['lvef'], nors_df['lvef'], equal_var=False)

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((n_dcm - 1) * np.var(dcms_df['lvef']) + (n_nor - 1) * np.var(nors_df['lvef'])) / (n_dcm + n_nor - 2))
    cohens_d = (np.mean(dcms_df['lvef']) - np.mean(nors_df['lvef'])) / pooled_std

    # Create visualization
    plt.figure(figsize=(10, 6))

    # Box plot
    data_for_plot = [dcms_df['lvef'], nors_df['lvef']]
    labels = ['DCM', 'NOR']

    plt.boxplot(data_for_plot, labels=labels)
    plt.ylabel('LVEF (%)')
    plt.title('LVEF Distribution by Group')
    plt.grid(True, alpha=0.3)

    # Add individual points
    for i, (group, data) in enumerate([('DCM', dcms_df['lvef']), ('NOR', nors_df['lvef'])]):
        plt.scatter([i+1] * len(data), data, alpha=0.6, label=group)

    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "lvef_group_comparison.png")
    plt.close()

    # Create violin plot as alternative
    plt.figure(figsize=(10, 6))
    df_plot = pd.DataFrame({
        'LVEF': list(dcms_df['lvef']) + list(nors_df['lvef']),
        'Group': ['DCM'] * len(dcms_df) + ['NOR'] * len(nors_df)
    })

    sns.violinplot(data=df_plot, x='Group', y='LVEF')
    plt.ylabel('LVEF (%)')
    plt.title('LVEF Distribution by Group (Violin Plot)')
    plt.tight_layout()
    plt.savefig(plots_dir / "lvef_group_comparison_violin.png")
    plt.close()

    # Save results to JSON
    statistical_results = {
        "analysis_type": "group_difference",
        "test_performed": "two-sample t-test",
        "p_value": float(p_value),
        "effect_size": float(cohens_d),
        "effect_size_type": "cohens_d",
        "n_total": int(n_dcm + n_nor),
        "sample_sizes": {
            "DCM": int(n_dcm),
            "NOR": int(n_nor)
        },
        "variables_tested": {
            "outcome": "LVEF",
            "predictors": []
        },
        "group_statistics": {
            "DCM": {
                "mean": float(np.mean(dcms_df['lvef'])),
                "std": float(np.std(dcms_df['lvef'])),
                "median": float(np.median(dcms_df['lvef'])),
                "min": float(np.min(dcms_df['lvef'])),
                "max": float(np.max(dcms_df['lvef']))
            },
            "NOR": {
                "mean": float(np.mean(nors_df['lvef'])),
                "std": float(np.std(nors_df['lvef'])),
                "median": float(np.median(nors_df['lvef'])),
                "min": float(np.min(nors_df['lvef'])),
                "max": float(np.max(nors_df['lvef']))
            }
        }
    }

    # Write to file
    with open(output_dir / "statistical_results.json", "w") as f:
        json.dump(statistical_results, f, indent=2)

    print("Analysis completed successfully!")
    print(f"DCM sample size: {n_dcm}")
    print(f"NOR sample size: {n_nor}")
    print(f"P-value: {p_value:.6f}")
    print(f"Cohen's d: {cohens_d:.4f}")

else:
    raise ValueError("One or both groups have insufficient sample sizes for analysis")