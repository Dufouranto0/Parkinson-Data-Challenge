#!/usr/bin/env python3

from pathlib import Path
import shutil
import pandas as pd
import nibabel as nib
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans


# ============================================================
# CONFIGURATION
# ============================================================

# Your current dataset
DATA = Path("../../data/train_set")

# Where the split dataset will be created
OUTPUT = Path("../../data/train_by_site")

# Number of sites requested
N_SITES = 9

# If True: create symbolic links instead of copying NIfTI files.
# This saves a huge amount of disk space.
USE_SYMLINKS = True

# Reproducibility
RANDOM_STATE = 42


# ============================================================
# LOAD LABELS
# ============================================================

labels_file = DATA / "train_labels.csv"
nifti_dir = DATA / "niftis"

labels = pd.read_csv(labels_file)

print(f"Number of subjects: {len(labels)}")


# ============================================================
# EXTRACT VOXEL SPACING
# ============================================================

print("\nReading NIfTI headers...")

records = []
missing = []

for uid in tqdm(labels["uid"]):

    nii_file = nifti_dir / f"{uid}.nii.gz"

    if not nii_file.exists():
        missing.append(uid)
        continue

    img = nib.load(str(nii_file))

    # First 3 values = voxel spacing in mm
    spacing = img.header.get_zooms()[:3]

    records.append({
        "uid": uid,
        "sx": float(spacing[0]),
        "sy": float(spacing[1]),
        "sz": float(spacing[2]),
    })


if missing:
    print(f"\nWARNING: {len(missing)} images are missing.")
    print(missing[:20])

spacing_df = pd.DataFrame(records)

print(f"\nImages with valid spacing: {len(spacing_df)}")


# ============================================================
# CLUSTER VOXEL SPACINGS
# ============================================================

X = spacing_df[["sx", "sy", "sz"]].values


print(f"\nClustering voxel spacings into {N_SITES} sites...")

kmeans = KMeans(
    n_clusters=N_SITES,
    random_state=RANDOM_STATE,
    n_init=50,
)

spacing_df["cluster"] = kmeans.fit_predict(X)


# ============================================================
# ORDER SITES BY THEIR VOXEL SPACING
#
# KMeans labels (0,1,2...) have no physical meaning.
# We reorder them so the site numbering is reproducible
# and easier to inspect.
# ============================================================

centers = kmeans.cluster_centers_

# Sort primarily by x spacing, then y, then z
order = sorted(
    range(N_SITES),
    key=lambda i: (
        centers[i][0],
        centers[i][1],
        centers[i][2],
    )
)

cluster_to_site = {
    cluster: site_number
    for site_number, cluster in enumerate(order, start=1)
}

spacing_df["site"] = spacing_df["cluster"].map(cluster_to_site)


# ============================================================
# PRINT SITE INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("DISCOVERED SITES")
print("=" * 80)

for site in range(1, N_SITES + 1):

    site_data = spacing_df[spacing_df["site"] == site]

    cluster = site_data["cluster"].iloc[0]

    center = centers[cluster]

    print(
        f"\nSITE {site:02d}"
        f" | N = {len(site_data):4d}"
        f" | centroid = "
        f"({center[0]:.5f}, "
        f"{center[1]:.5f}, "
        f"{center[2]:.5f})"
    )

    # Show the most common exact spacings in this site
    counts = (
        site_data[["sx", "sy", "sz"]]
        .round(5)
        .value_counts()
        .head(10)
    )

    for spacing, count in counts.items():
        print(
            f"    {spacing} : {count}"
        )


# ============================================================
# MERGE WITH LABELS
# ============================================================

result = labels.merge(
    spacing_df[["uid", "sx", "sy", "sz", "site"]],
    on="uid",
    how="left",
)

if result["site"].isna().any():
    print("\nWARNING: Some subjects have no site assignment.")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# SAVE GLOBAL SITE ASSIGNMENT FILE
# ============================================================

assignment_file = OUTPUT / "site_assignments.csv"

result.to_csv(
    assignment_file,
    index=False,
)

print(f"\nSaved: {assignment_file}")


# ============================================================
# CREATE SITE DIRECTORIES
# ============================================================

for site in range(1, N_SITES + 1):

    site_dir = OUTPUT / f"site_{site:02d}"
    site_nifti_dir = site_dir / "niftis"

    site_nifti_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    site_data = result[result["site"] == site].copy()

    # Save labels for this site
    site_labels = site_data[
        ["uid", "is_pathologic"]
    ]

    site_labels.to_csv(
        site_dir / "train_labels.csv",
        index=False,
    )

    print(
        f"\nCreating site_{site:02d}: "
        f"{len(site_data)} subjects"
    )

    # --------------------------------------------------------
    # Link/copy NIfTI files
    # --------------------------------------------------------

    for uid in tqdm(
        site_data["uid"],
        desc=f"site_{site:02d}",
    ):

        source = nifti_dir / f"{uid}.nii.gz"
        destination = site_nifti_dir / f"{uid}.nii.gz"

        if destination.exists():
            continue

        if USE_SYMLINKS:
            destination.symlink_to(
                source.resolve()
            )
        else:
            shutil.copy2(
                source,
                destination,
            )


# ============================================================
# SAVE A HUMAN-READABLE SITE SUMMARY
# ============================================================

summary_rows = []

for site in range(1, N_SITES + 1):

    site_data = result[result["site"] == site]

    if len(site_data) == 0:
        continue

    summary_rows.append({
        "site": site,
        "n_subjects": len(site_data),
        "mean_sx": site_data["sx"].mean(),
        "mean_sy": site_data["sy"].mean(),
        "mean_sz": site_data["sz"].mean(),
        "min_sx": site_data["sx"].min(),
        "max_sx": site_data["sx"].max(),
        "min_sy": site_data["sy"].min(),
        "max_sy": site_data["sy"].max(),
        "min_sz": site_data["sz"].min(),
        "max_sz": site_data["sz"].max(),
        "pathologic": (
            site_data["is_pathologic"] == 1
        ).sum(),
        "non_pathologic": (
            site_data["is_pathologic"] == 0
        ).sum(),
    })


summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUTPUT / "site_summary.csv",
    index=False,
)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(f"\nOutput directory:")
print(OUTPUT.resolve())

print("\nSite summary:")
print(summary.to_string(index=False))

print("\nFiles created:")
print(f"  {OUTPUT}/site_01/")
print(f"  {OUTPUT}/site_02/")
print(f"  ...")
print(f"  {OUTPUT}/site_{N_SITES:02d}/")
print(f"  {OUTPUT}/site_assignments.csv")
print(f"  {OUTPUT}/site_summary.csv")

if USE_SYMLINKS:
    print(
        "\nNIfTI files were SYMBOLICALLY LINKED, "
        "not copied. Original data remain untouched."
    )
else:
    print(
        "\nNIfTI files were COPIED."
        )

