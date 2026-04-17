# filename: phase2b_lvef_welch_ttest.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

results_db_path = "$BIO_DATA_ROOT/sat_cache/acdc"

groups_of_interest = ["DCM", "NOR"]
structure = "left heart ventricle"
observations = ["ED", "ES"]

out_data_dir = Path("data")
out_plots_dir = Path("plots")
out_data_dir.mkdir(parents=True, exist_ok=True)
out_plots_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# 1) Load cohort + compute LVEDV/LVESV/LVEF
# -------------------------
patient_ids = sat.list_patients(results_db_path)

rows = []
exclusions = []

def volume_from_masks(mask_dicts):
    """Sum volumes across mask list entries."""
    if mask_dicts is None or len(mask_dicts) == 0:
        return None
    vols = []
    for d in mask_dicts:
        if "mask" not in d or "spacing" not in d:
            continue
        try:
            v = sat.calculate_volume(d["mask"], d["spacing"])
            if np.isfinite(v):
                vols.append(float(v))
        except Exception:
            continue
    if len(vols) == 0:
        return None
    return float(np.sum(vols))

for pid in patient_ids:
    try:
        md = sat.get_patient_metadata(pid)
    except Exception as e:
        exclusions.append({"patient_id": pid, "reason": f"metadata_load_failed: {type(e).__name__}: {e}"})
        continue

    grp = md.get("group", None)
    if grp not in groups_of_interest:
        continue

    # observation identifier mapping (ED/ES -> identifier strings)
    try:
        obs_map = sat.get_observation_identifiers(pid)
    except Exception as e:
        exclusions.append({"patient_id": pid, "group": grp, "reason": f"observation_identifier_load_failed: {type(e).__name__}: {e}"})
        continue

    if not isinstance(obs_map, dict) or any(o not in obs_map for o in observations):
        exclusions.append({"patient_id": pid, "group": grp, "reason": f"missing_ED_or_ES_identifier: have={list(obs_map.keys()) if isinstance(obs_map, dict) else type(obs_map)}"})
        continue

    vols = {}
    vol_fail = False
    for obs in observations:
        identifier = obs_map[obs]
        try:
            mask_dicts = sat.load_structure_mask(
                results_db_path,
                pid,
                structure,
                source_image_contains=identifier
            )
        except Exception as e:
            exclusions.append({"patient_id": pid, "group": grp, "reason": f"mask_load_failed_{obs}: {type(e).__name__}: {e}"})
            vol_fail = True
            break

        vol = volume_from_masks(mask_dicts)
        if vol is None:
            exclusions.append({"patient_id": pid, "group": grp, "reason": f"volume_compute_failed_or_empty_mask_{obs}"})
            vol_fail = True
            break

        vols[obs] = vol

    if vol_fail:
        continue

    edv = vols["ED"]
    esv = vols["ES"]

    # Basic QC: positive EDV and nonnegative ESV; EDV should exceed ESV for meaningful EF
    if not (np.isfinite(edv) and np.isfinite(esv)):
        exclusions.append({"patient_id": pid, "group": grp, "reason": "nonfinite_volume"})
        continue
    if edv <= 0:
        exclusions.append({"patient_id": pid, "group": grp, "reason": f"invalid_EDV<=0: {edv}"})
        continue
    if esv < 0:
        exclusions.append({"patient_id": pid, "group": grp, "reason": f"invalid_ESV<0: {esv}"})
        continue

    lvef = (edv - esv) / edv * 100.0

    if not np.isfinite(lvef):
        exclusions.append({"patient_id": pid, "group": grp, "reason": "nonfinite_LVEF"})
        continue

    rows.append({
        "patient_id": pid,
        "group": grp,
        "LVEDV_mL": float(edv),
        "LVESV_mL": float(esv),
        "LVEF_percent": float(lvef),
        # keep allowed metadata fields (not used for modeling per plan)
        "height": md.get("height", None),
        "weight": md.get("weight", None),
        "ed_frame": md.get("ed_frame", None),
        "es_frame": md.get("es_frame", None),
        "num_frames": md.get("num_frames", None),
    })

df = pd.DataFrame(rows)

# Ensure both groups have data
sample_sizes = df["group"].value_counts().to_dict()
for g in groups_of_interest:
    if sample_sizes.get(g, 0) == 0:
        raise RuntimeError(f"Planned group '{g}' has n=0 after loading/QC. sample_sizes={sample_sizes}")

# -------------------------
# 2) Primary test: Welch two-sample t-test (two-sided)
# Convention: group1=DCM, group2=NOR; positive effect => DCM higher than NOR
# -------------------------
g1, g2 = "DCM", "NOR"
x1 = df.loc[df["group"] == g1, "LVEF_percent"].astype(float).to_numpy()
x2 = df.loc[df["group"] == g2, "LVEF_percent"].astype(float).to_numpy()

tt = stats.ttest_ind(x1, x2, equal_var=False, alternative="two-sided")
p_value = float(tt.pvalue)

mean1, mean2 = float(np.mean(x1)), float(np.mean(x2))
std1, std2 = float(np.std(x1, ddof=1)), float(np.std(x2, ddof=1))
n1, n2 = int(len(x1)), int(len(x2))
diff = mean1 - mean2  # DCM - NOR (positive => DCM higher)

# Cohen's d with pooled SD (per spec)
pooled_sd = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
cohens_d = float(diff / pooled_sd) if pooled_sd > 0 else np.nan

# 95% CI for mean difference using Welch-Satterthwaite df
se = np.sqrt(std1**2 / n1 + std2**2 / n2)
df_welch = (std1**2 / n1 + std2**2 / n2) ** 2 / ((std1**2 / n1) ** 2 / (n1 - 1) + (std2**2 / n2) ** 2 / (n2 - 1))
tcrit = stats.t.ppf(0.975, df_welch)
ci_low = float(diff - tcrit * se)
ci_high = float(diff + tcrit * se)

# Per-group summary stats for output
def summarize(arr):
    arr = np.asarray(arr, dtype=float)
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
        "median": float(np.median(arr)),
        "iqr": float(q3 - q1),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

group_statistics = {
    g1: summarize(x1),
    g2: summarize(x2),
}

# -------------------------
# 3) Visualization
# -------------------------
plt.figure(figsize=(7.2, 4.8), dpi=160)
order = [g2, g1]  # show NOR then DCM
sns.boxplot(data=df, x="group", y="LVEF_percent", order=order, showfliers=False, color="#d9d9d9")
sns.stripplot(data=df, x="group", y="LVEF_percent", order=order, alpha=0.75, size=4)
plt.title("LVEF (%) by Group (NOR vs DCM)")
plt.xlabel("Group")
plt.ylabel("LVEF (%)")
plt.tight_layout()
plot_path = out_plots_dir / "lvef_group_comparison.png"
plt.savefig(plot_path)
plt.close()

# -------------------------
# 4) Save outputs
# -------------------------
results = {
    "analysis_type": "group_difference",
    "test_performed": "Welch two-sample t-test (independent, two-sided)",
    "p_value": p_value,
    "effect_size": cohens_d,
    "effect_size_type": "cohens_d",
    "n_total": int(df.shape[0]),
    "sample_sizes": {k: int(v) for k, v in sample_sizes.items() if k in groups_of_interest},
    "variables_tested": {"outcome": "LVEF_percent", "predictors": ["group (DCM vs NOR)"]},
    "group_statistics": group_statistics,
    "test_details": {
        "group1": g1,
        "group2": g2,
        "mean_difference_group1_minus_group2": float(diff),
        "ci95_mean_difference": [ci_low, ci_high],
        "t_statistic": float(tt.statistic),
        "welch_df": float(df_welch),
    },
    "data_provenance": {
        "results_db_path": results_db_path,
        "structures": [structure],
        "observations": observations,
        "metric_definition": "LVEF = ((LVEDV - LVESV) / LVEDV) * 100",
        "volume_definition": "sum over mask entries of sat.calculate_volume(mask, spacing)",
    },
    "qc": {
        "n_excluded": int(len(exclusions)),
        "exclusions": exclusions[:50],  # truncate for file size; full can be saved separately if needed
        "exclusions_truncated": len(exclusions) > 50,
    },
    "artifacts": {
        "plot_paths": [str(plot_path)],
    }
}

out_path = out_data_dir / "statistical_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

# Also save the computed per-patient table for traceability (allowed; not requested but useful)
df_out_path = out_data_dir / "derived_lvef_table.csv"
df.to_csv(df_out_path, index=False)

print("Done.")
print("Included sample sizes:", results["sample_sizes"])
print("Welch t-test p =", results["p_value"])
print("Mean(LVEF) DCM =", group_statistics[g1]["mean"], "NOR =", group_statistics[g2]["mean"])
print("Mean diff (DCM-NOR) =", diff, "95% CI =", (ci_low, ci_high))
print("Cohen's d =", cohens_d)
print("Wrote:", out_path)
print("Plot:", plot_path)
print("Derived table:", df_out_path)