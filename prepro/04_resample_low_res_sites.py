#!/usr/bin/env python3
"""
Génère, pour les sujets acquis à haute résolution native, des copies "dégradées"
au niveau des résolutions des sites peu représentés du train set, afin de
rééquilibrer la distribution résolution/site vue par le modèle.

Contexte / pourquoi ce script existe
-------------------------------------
`preprocess_dataset.py` rééchantillonne TOUJOURS chaque examen vers la même
grille finale (DEFAULT_TARGET_SPACING, ex: 1.5mm iso). Un examen natif à
1.2mm est donc *légèrement sous-échantillonné* pour atteindre 1.5mm, tandis
qu'un examen natif à 3.3mm ou 4.1mm est *fortement suréchantillonné* (BSpline)
pour atteindre la même grille. Ces deux opérations laissent des signatures
fréquentielles très différentes dans le volume final (flou d'interpolation,
absence de haute fréquence réelle pour les gros voxels natifs, etc.), et cette
signature est corrélée avec le site d'acquisition -- donc avec des facteurs de
confusion (scanner, protocole) plutôt qu'avec la pathologie. Un modèle peut
apprendre ce raccourci, ce qui casse la généralisation vers des sites peu vus
en train.

Ce script attaque le problème *en amont* du pipeline existant : pour chaque
sujet dont la résolution native est strictement meilleure que 2.5mm iso sur
les 3 axes, on fabrique des NIfTI synthétiques rééchantillonnés (downsampling
uniquement, jamais d'upsampling) vers les résolutions natives des sites peu
représentés. Ces nouveaux fichiers sont ensuite traités par
`preprocess_dataset.py` exactement comme n'importe quel examen natif -- on ne
duplique aucune logique de crop/normalisation, on ne fait que produire de
nouveaux ".nii.gz" d'entrée qui, une fois passés dans le pipeline existant,
porteront la même "signature de suréchantillonnage" qu'un vrai examen natif
à cette résolution, mais avec un label et une anatomie connus et variés.

Anti-repli garanti
-------------------
- On ne descend JAMAIS en dessous de la résolution native d'un sujet : une
  cible n'est retenue pour un sujet donné que si elle est strictement plus
  grossière que sa résolution native sur les 3 axes (avec une marge
  `--min-downsample-ratio` pour éviter les rééchantillonnages quasi no-op).
- Anti-aliasing : un lissage gaussien (sigma dépendant du spacing cible) est
  appliqué avant le rééchantillonnage sur les axes réellement sous-échantillonnés,
  pour éviter le repliement de spectre (moiré / crénelage).

Anti-fuite (leakage) entre splits
-----------------------------------
Un même sujet peut désormais exister sous plusieurs "uid" (uid natif +
plusieurs uid synthétiques dérivés). Le CSV de sortie porte donc une colonne
`source_uid` (= uid du sujet réel dont dérive chaque ligne, y compris pour les
lignes réelles où `source_uid == uid`). Utiliser `get_stratified_group_splits`
(fourni dans ce fichier, à copier/fusionner dans `dataloader.py`) qui
groupe explicitement sur `source_uid` via `StratifiedGroupKFold` : toutes les
variantes d'un même sujet tombent alors forcément dans le même fold.

Usage typique
-------------
    python resample_low_res_sites.py \\
        --input-dir /neurospin/tmp/ad279118/data_ad279118_pd/train_set/niftis \\
        --site-assignments /neurospin/tmp/ad279118/data_ad279118_pd/train_set/site_assignments.csv \\
        --output-dir resample_train_set \\
        --better-than 2.5 2.5 2.5 \\
        --jobs 8

Puis, pour l'entraînement, fusionner les deux CSV (le script écrit déjà les
deux blocs -- réel + synthétique -- dans un seul fichier de sortie) et pointer
`DaTscanDataset`/`preprocess_dataset.py` vers l'union des deux dossiers de
NIfTI (natif + `resample_train_set/niftis`).
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from loguru import logger
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm


# =============================================================================
# Utilitaires génériques (cohérents avec preprocess_dataset.py)
# =============================================================================

def _uid_from_path(p: Path) -> str:
    """Même convention que preprocess_dataset._uid_from_path : on la duplique
    ici pour que ce script reste autonome (pas de dépendance d'import sur la
    structure du package `my_module`)."""
    name = p.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return p.stem


def _format_spacing_suffix(spacing: tuple) -> str:
    """ex: (3.3, 3.3, 1.0) -> 'res3p30x3p30x1p00' (pas de '.' dans un nom de
    fichier, pour éviter toute confusion avec l'extension .nii.gz)."""
    parts = [f"{v:.2f}".replace(".", "p") for v in spacing]
    return "res" + "x".join(parts)


def read_spacing_header_only(path: Path) -> tuple:
    """Lit uniquement les métadonnées (spacing) d'un NIfTI, sans charger les
    voxels : nettement plus rapide qu'un sitk.ReadImage complet quand on ne
    fait que scanner les résolutions de centaines d'examens."""
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return tuple(float(s) for s in reader.GetSpacing())


# =============================================================================
# Étape 1 : scan des résolutions natives
# =============================================================================

def scan_native_resolutions(input_dir: Path, show_progress: bool = True) -> pd.DataFrame:
    """Parcourt tous les NIfTI de input_dir, lit le spacing de chacun (header
    seul) et retourne un DataFrame [uid, sx, sy, sz, filepath].

    C'est la boucle demandée : "charger les niftis un par un pour connaître
    leur résolution" -- une fois que TOUS les fichiers ont été vus, on peut
    déterminer les résolutions cibles (celles des sites peu représentés) et
    seulement ensuite décider qui rééchantillonner vers quoi.
    """
    files = sorted(
        p for p in input_dir.glob("**/*") if p.is_file() and p.name.endswith((".nii", ".nii.gz"))
    )
    if not files:
        raise FileNotFoundError(f"Aucun NIfTI trouvé dans {input_dir}")

    rows = []
    iterator = tqdm(files, desc="Lecture des résolutions natives") if show_progress else files
    for f in iterator:
        try:
            sx, sy, sz = read_spacing_header_only(f)
        except Exception as e:
            logger.error(f"Impossible de lire le header de {f.name} : {e}")
            continue
        rows.append({"uid": _uid_from_path(f), "sx": sx, "sy": sy, "sz": sz, "filepath": str(f)})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Aucune résolution n'a pu être lue -- vérifier input_dir.")
    return df


# =============================================================================
# Étape 2 : détermination des résolutions cibles (sites peu représentés)
# =============================================================================

@dataclass
class TargetSpec:
    site_id: str          # identifiant du site "peu représenté" d'origine
    spacing: tuple         # (sx, sy, sz) cible, résolution médiane réelle de ce site
    n_native_subjects: int  # nombre de sujets réels qui définissent ce site


def determine_small_site_targets(
    site_df: pd.DataFrame,
    small_site_threshold: Optional[int] = 100,
    min_target_count: int = 10,
    round_decimals: int = 2,
) -> list:
    """À partir du CSV de sites (colonnes: uid, is_pathologic, sx, sy, sz, site),
    identifie les sites "peu représentés" et retourne, pour chacun, la
    résolution cible à utiliser pour le rééchantillonnage synthétique
    (médiane des spacings réels des sujets de ce site -- plus robuste qu'une
    moyenne face à un site hétérogène ou à un outlier).

    `small_site_threshold` : nombre de sujets en-dessous duquel un site est
    considéré comme "peu représenté". Si None (défaut), on prend la médiane
    des effectifs par site -- les sites en dessous de la médiane sont "petits".

    `min_target_count` : on écarte les sites avec moins de sujets que ce seuil
    de la liste des CIBLES (même s'ils sont "petits") : un site à 1 ou 2
    sujets est trop peu fiable pour en déduire une résolution cible
    représentative -- ce serait probablement un outlier/artefact plutôt qu'une
    vraie résolution de site à rééquilibrer.
    """
    counts = site_df["site"].value_counts()
    threshold = small_site_threshold if small_site_threshold is not None else counts.median()

    logger.info(f"Effectifs par site :\n{counts.sort_index()}")
    logger.info(f"Seuil 'site peu représenté' (nb sujets) : < {threshold}")

    small_sites = counts[counts < threshold].index.tolist()

    targets = []
    seen_spacings = set()
    for site_id in small_sites:
        sub = site_df[site_df["site"] == site_id]
        n = len(sub)
        if n < min_target_count:
            logger.warning(
                f"Site {site_id!r} ignoré comme cible ({n} sujet(s) < min_target_count="
                f"{min_target_count}) : trop peu de sujets pour une résolution cible fiable."
            )
            continue

        spacing = tuple(round(float(sub[c].median()), round_decimals) for c in ("sx", "sy", "sz"))

        # Déduplication : deux petits sites peuvent correspondre à la même
        # résolution physique (arrondie) -- inutile de dupliquer le travail.
        if spacing in seen_spacings:
            logger.info(f"Site {site_id!r} : résolution cible {spacing} déjà couverte, ignoré.")
            continue
        seen_spacings.add(spacing)

        targets.append(TargetSpec(site_id=str(site_id), spacing=spacing, n_native_subjects=n))
        logger.info(f"Cible retenue -> site {site_id!r} : spacing médian {spacing} ({n} sujets natifs)")

    if not targets:
        logger.warning("Aucune résolution cible retenue : rien à générer.")
    return targets


# =============================================================================
# Étape 3 : rééchantillonnage physique (downsampling uniquement)
# =============================================================================

def resample_to_spacing(img: sitk.Image, target_spacing: tuple) -> sitk.Image:
    """Rééchantillonne `img` vers `target_spacing`, en préservant l'étendue
    physique du volume (même origine/direction, taille recalculée). Un
    lissage gaussien anti-repliement est appliqué au préalable sur les axes
    effectivement sous-échantillonnés (sigma ~ moitié du voxel cible, dans le
    même esprit que le lissage physique déjà utilisé dans
    preprocess_dataset.py).
    """
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    sigma_mm = [
        (t / 2.0) if t > o * 1.001 else 0.0
        for o, t in zip(original_spacing, target_spacing)
    ]
    if any(s > 0 for s in sigma_mm):
        img = sitk.SmoothingRecursiveGaussian(img, sigma_mm, normalizeAcrossScale=False)

    new_size = [
        max(1, int(round(osz * ospc / tspc)))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resampled = sitk.Resample(
        img,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        img.GetOrigin(),
        target_spacing,
        img.GetDirection(),
        0.0,
        img.GetPixelID(),
    )
    return resampled


def _process_one(job: dict) -> dict:
    """Exécuté dans un worker séparé (ProcessPoolExecutor). Retourne un dict
    décrivant le résultat (succès + métadonnées) plutôt que de laisser une
    exception individuelle interrompre tout le run -- même logique défensive
    que `process_single_file` dans preprocess_dataset.py."""
    try:
        img = sitk.ReadImage(job["filepath"], sitk.sitkFloat32)
        resampled = resample_to_spacing(img, job["target_spacing"])
        out_path = Path(job["out_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(resampled, str(out_path))
        return {**job, "ok": True, "error": None}
    except Exception as e:
        return {**job, "ok": False, "error": str(e)}


def build_jobs(
    native_df: pd.DataFrame,
    targets: list,
    better_than: tuple,
    min_downsample_ratio: float,
    out_niftis_dir: Path,
) -> list:
    """Construit la liste des jobs de rééchantillonnage : un job par (sujet
    éligible x résolution cible compatible). Une cible n'est retenue pour un
    sujet que si elle est strictement plus grossière que sa résolution native
    sur LES TROIS axes (jamais d'upsampling, jamais de mélange
    downsample-un-axe / upsample-un-autre)."""
    bx, by, bz = better_than
    eligible = native_df[(native_df["sx"] < bx) & (native_df["sy"] < by) & (native_df["sz"] < bz)]
    logger.info(
        f"{len(eligible)}/{len(native_df)} sujets ont une résolution native meilleure que "
        f"{better_than} sur les 3 axes -> éligibles à la dégradation synthétique."
    )

    jobs = []
    for _, row in eligible.iterrows():
        native_spacing = (row["sx"], row["sy"], row["sz"])
        for t in targets:
            coarser_enough = all(
                tv >= nv * min_downsample_ratio for tv, nv in zip(t.spacing, native_spacing)
            )
            if not coarser_enough:
                continue
            new_uid = f"{row['uid']}_{_format_spacing_suffix(t.spacing)}"
            out_path = out_niftis_dir / f"{new_uid}.nii.gz"
            jobs.append(
                {
                    "source_uid": row["uid"],
                    "new_uid": new_uid,
                    "filepath": row["filepath"],
                    "native_spacing": native_spacing,
                    "target_spacing": t.spacing,
                    "target_site": t.site_id,
                    "out_path": str(out_path),
                }
            )
    logger.info(f"{len(jobs)} rééchantillonnages synthétiques à générer.")
    return jobs


def run_jobs(jobs: list, jobs_workers: int, show_progress: bool = True) -> list:
    results = []
    if not jobs:
        return results
    with ProcessPoolExecutor(max_workers=jobs_workers) as executor:
        futures = {executor.submit(_process_one, job): job["new_uid"] for job in jobs}
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="Rééchantillonnage synthétique")
        for future in iterator:
            new_uid = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Exception non interceptée pour {new_uid} : {e}")
                results.append({"new_uid": new_uid, "ok": False, "error": str(e)})
    return results


# =============================================================================
# Étape 4 : assemblage du site_assignments.csv de sortie (réel + synthétique)
# =============================================================================

def build_output_csv(
    site_df: pd.DataFrame,
    results: list,
) -> pd.DataFrame:
    """Construit le CSV final : toutes les lignes réelles (avec source_uid ==
    uid, is_synthetic=False) + toutes les lignes synthétiques réussies (avec
    source_uid == uid du sujet réel dont elles dérivent, is_synthetic=True).
    Le label `is_pathologic` d'une ligne synthétique est toujours celui du
    sujet réel source -- c'est bien le même patient, seule la résolution
    change."""
    real_df = site_df.copy()
    real_df["source_uid"] = real_df["uid"]
    real_df["is_synthetic"] = False
    real_df = real_df[["uid", "is_pathologic", "sx", "sy", "sz", "site", "source_uid", "is_synthetic"]]

    label_by_uid = site_df.set_index("uid")["is_pathologic"].to_dict()

    synth_rows = []
    n_failed = 0
    for r in results:
        if not r.get("ok"):
            n_failed += 1
            logger.error(f"Échec pour {r.get('new_uid')} (source={r.get('source_uid')}) : {r.get('error')}")
            continue
        synth_rows.append(
            {
                "uid": r["new_uid"],
                "is_pathologic": label_by_uid.get(r["source_uid"]),
                "sx": r["target_spacing"][0],
                "sy": r["target_spacing"][1],
                "sz": r["target_spacing"][2],
                "site": r["target_site"],
                "source_uid": r["source_uid"],
                "is_synthetic": True,
            }
        )
    if n_failed:
        logger.warning(f"{n_failed} rééchantillonnage(s) ont échoué et sont exclus du CSV de sortie.")

    synth_df = pd.DataFrame(synth_rows)
    out_df = pd.concat([real_df, synth_df], ignore_index=True) if not synth_df.empty else real_df
    return out_df


# =============================================================================
# Splits stratifiés + groupés (anti-fuite) -- à fusionner dans dataloader.py
# =============================================================================

def get_stratified_group_splits(
    labels_csv: Path,
    n_splits: int = 5,
    seed: int = 42,
    group_col: str = "source_uid",
):
    """Équivalent de `get_stratified_splits` (dataloader.py) mais groupé : deux
    lignes qui partagent le même `group_col` (typiquement `source_uid`, donc
    un sujet réel + toutes ses variantes de résolution synthétiques) tombent
    TOUJOURS dans le même fold. Sans ça, une fuite se produirait dès qu'un
    sujet dont une variante synthétique atterrit en train alors que le sujet
    natif (ou une autre variante) est en validation : le modèle aurait vu la
    même anatomie/le même bruit anatomique des deux côtés du split.

    Si `group_col` est absent du CSV (cas d'un site_assignments.csv "classique"
    sans rééchantillonnage synthétique), se comporte comme un split stratifié
    simple (chaque sujet est son propre groupe).
    """
    df = pd.read_csv(labels_csv)
    if "site" not in df.columns:
        raise ValueError("Le fichier CSV doit contenir une colonne 'site' pour la stratification des centres.")
    if group_col not in df.columns:
        df[group_col] = df["uid"]

    df["strat_key"] = df["site"].astype(str) + "_" + df["is_pathologic"].astype(str)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(sgkf.split(df, df["strat_key"], groups=df[group_col]))

    # Vérification défensive : aucun groupe ne doit apparaître des deux côtés
    # d'un même fold.
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        train_groups = set(df.iloc[train_idx][group_col])
        val_groups = set(df.iloc[val_idx][group_col])
        overlap = train_groups & val_groups
        if overlap:
            raise RuntimeError(f"Fuite détectée au fold {fold_i} : groupes partagés {overlap}")

    return df, splits


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Génère des NIfTI synthétiques dégradés vers les résolutions des sites peu représentés."
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Dossier des NIfTI natifs du train set.")
    parser.add_argument("--site-assignments", type=str, required=True, help="CSV existant (uid,is_pathologic,sx,sy,sz,site).")
    parser.add_argument("--output-dir", type=str, default="resample_train_set", help="Dossier de sortie.")
    parser.add_argument("--better-than", type=float, nargs=3, default=[2.5, 2.5, 2.5], help="Seuil de résolution native (mm) à ne dépasser sur AUCUN des 3 axes pour être éligible à la dégradation.")
    parser.add_argument("--small-site-threshold", type=int, default=100, help="Nb de sujets en dessous duquel un site est 'peu représenté' (défaut: médiane des effectifs par site).")
    parser.add_argument("--min-target-count", type=int, default=4, help="Nb minimum de sujets réels requis dans un petit site pour en faire une résolution cible fiable.")
    parser.add_argument("--min-downsample-ratio", type=float, default=1.05, help="Marge minimale (cible/natif) exigée sur chaque axe pour retenir une cible -- évite les rééchantillonnages quasi no-op.")
    parser.add_argument("--jobs", type=int, default=4, help="Nombre de processus parallèles pour le rééchantillonnage.")
    parser.add_argument("--dry-run", action="store_true", help="N'affiche que les résolutions cibles et le nombre de jobs, sans écrire de fichiers.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    out_niftis_dir = output_dir / "niftis"
    site_csv_path = Path(args.site_assignments)

    site_df = pd.read_csv(site_csv_path)
    required_cols = {"uid", "is_pathologic", "sx", "sy", "sz", "site"}
    missing = required_cols - set(site_df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {site_csv_path} : {missing}")

    # --- Étape 1 : scan des résolutions natives directement depuis les NIfTI ---
    native_df = scan_native_resolutions(input_dir)

    merged = native_df.merge(site_df[["uid", "is_pathologic", "site"]], on="uid", how="inner")
    n_unmatched = len(native_df) - len(merged)
    if n_unmatched:
        logger.warning(
            f"{n_unmatched} NIfTI de {input_dir} n'ont pas de correspondance dans "
            f"{site_csv_path} (uid absent) -- ignorés."
        )
    n_csv_only = len(site_df) - len(merged)
    if n_csv_only:
        logger.warning(
            f"{n_csv_only} lignes de {site_csv_path} n'ont pas de fichier NIfTI correspondant "
            f"dans {input_dir} -- ignorées pour le scan de résolution (mais conservées telles "
            f"quelles dans le CSV de sortie)."
        )

    # Sanity check : la résolution relue depuis le header doit être cohérente
    # avec celle déjà présente dans site_assignments.csv.
    check = merged.merge(site_df[["uid", "sx", "sy", "sz"]], on="uid", suffixes=("", "_csv"))
    drift = check[
        (np.abs(check["sx"] - check["sx_csv"]) > 0.05)
        | (np.abs(check["sy"] - check["sy_csv"]) > 0.05)
        | (np.abs(check["sz"] - check["sz_csv"]) > 0.05)
    ]
    if not drift.empty:
        logger.warning(
            f"{len(drift)} sujet(s) ont une résolution relue différente de celle du CSV "
            f"(> 0.05mm d'écart) -- le scan des fichiers fait foi pour le rééchantillonnage :\n"
            f"{drift[['uid', 'sx', 'sy', 'sz', 'sx_csv', 'sy_csv', 'sz_csv']].to_string(index=False)}"
        )

    # --- Étape 2 : résolutions cibles à partir des sites peu représentés ---
    targets = determine_small_site_targets(
        site_df,
        small_site_threshold=args.small_site_threshold,
        min_target_count=args.min_target_count,
    )
    if not targets:
        logger.warning("Rien à générer, arrêt.")
        return

    # --- Étape 3 : construction + exécution des jobs de rééchantillonnage ---
    jobs = build_jobs(
        native_df=merged,
        targets=targets,
        better_than=tuple(args.better_than),
        min_downsample_ratio=args.min_downsample_ratio,
        out_niftis_dir=out_niftis_dir,
    )

    if args.dry_run:
        logger.info("--dry-run : aucun fichier ne sera écrit.")
        by_target = pd.DataFrame(jobs)
        if not by_target.empty:
            logger.info(f"Répartition des jobs par site cible :\n{by_target['target_site'].value_counts()}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    results = run_jobs(jobs, jobs_workers=args.jobs)

    # --- Étape 4 : CSV de sortie (réel + synthétique) ---
    out_df = build_output_csv(site_df, results)
    out_csv_path = output_dir / "site_assignments.csv"
    out_df.to_csv(out_csv_path, index=False)

    n_ok = sum(1 for r in results if r.get("ok"))
    logger.success(
        f"Terminé. {n_ok}/{len(results)} NIfTI synthétiques écrits dans {out_niftis_dir}. "
        f"CSV combiné (réel + synthétique) : {out_csv_path} ({len(out_df)} lignes)."
    )
    logger.info(
        "Pour l'entraînement : concaténer les dossiers de NIfTI natifs et "
        f"{out_niftis_dir} avant de lancer preprocess_dataset.py, et utiliser "
        "get_stratified_group_splits (group_col='source_uid') plutôt que "
        "get_stratified_splits pour construire les folds, afin qu'un même sujet "
        "(natif + variantes synthétiques) reste toujours dans un seul et même fold."
    )


if __name__ == "__main__":
    main()
