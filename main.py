import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from module.model import build_model, probs_from_logits
from module.preprocess_dataset import preprocess_files, DEFAULT_CROP_SIZE, DEFAULT_TARGET_SPACING

DATA_DIR = Path("data")
NIFTIS_DIR = DATA_DIR / "niftis"
SUBMISSION_FORMAT_CSV = DATA_DIR / "submission_format.csv"
OUTPUT_CSV = Path("submission.csv")
RESULTS_DIR = Path("results")

# Écrit par train.py à partir de la prévalence globale du train (voir train.py).
# Important : dans le container de soumission, data/ ne contient QUE
# submission_format.csv et les NIfTI de test — aucun fichier de labels (ni train, ni
# test) n'y est présent. La prédiction de repli ne peut donc pas être recalculée ici
# et doit avoir été précalculée à l'entraînement puis embarquée dans results/.
FALLBACK_PROBABILITY_PATH = RESULTS_DIR / "fallback_probability.json"

PREPROCESS_JOBS = 4


class InferenceDataset(Dataset):
    def __init__(self, uids, cache_dir: Path):
        self.uids = uids
        self.cache_dir = Path(cache_dir)

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = str(self.uids[idx])
        npy_path = self.cache_dir / f"{uid}.npy"
        try:
            arr = np.load(npy_path)
        except Exception:
            # Filet de sécurité : les uids passés ici ont déjà réussi le
            # prétraitement (voir run_preprocessing), mais on se protège aussi
            # contre un fichier corrompu/manquant de dernière minute. Aucun détail
            # (uid, chemin, erreur) n'est loggé ici — voir la règle "no logging of
            # test dataset information". Le uid sera simplement absent des
            # prédictions du modèle et recevra la prédiction de repli à l'assemblage
            # final.
            return None

        tensor = torch.from_numpy(arr).float().unsqueeze(0)
        return tensor, uid


def collate_skip_none(batch):
    """Ignore les échantillons en échec (voir InferenceDataset.__getitem__)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    tensors, uids = zip(*batch)
    return torch.stack(tensors), list(uids)


def get_default_probability() -> float:
    """Prédiction de repli utilisée pour tout examen dont le prétraitement ou le
    chargement échoue : la prévalence globale du train, précalculée par train.py
    (voir FALLBACK_PROBABILITY_PATH ci-dessus). Repli sur 0.5 si le fichier est
    absent du bundle de soumission (ne devrait jamais arriver en usage normal).
    """
    if FALLBACK_PROBABILITY_PATH.exists():
        try:
            with open(FALLBACK_PROBABILITY_PATH) as f:
                return float(json.load(f)["fallback_probability"])
        except Exception as e:
            logger.warning(f"{FALLBACK_PROBABILITY_PATH} illisible ({e}). Repli sur 0.5.")
            return 0.5
    logger.warning(f"{FALLBACK_PROBABILITY_PATH} introuvable dans le bundle de soumission. Repli sur 0.5.")
    return 0.5


def resolve_preprocessing_params(fold_configs: list):
    """Détermine crop_size/target_spacing à partir des config.json des folds
    (écrits par train.py) plutôt que de les dupliquer en dur ici — évite toute
    dérive entre le prétraitement utilisé à l'entraînement et celui utilisé à
    l'inférence.

    Lève une erreur si les folds déclarent des valeurs incohérentes entre eux (bug
    de packaging à corriger avant soumission). Se replie sur les défauts du module
    de prétraitement si un fold plus ancien ne contient pas encore ces clés.
    """
    crop_sizes = {tuple(c["crop_size"]) for c in fold_configs if "crop_size" in c}
    spacings = {tuple(c["target_spacing"]) for c in fold_configs if "target_spacing" in c}

    if len(crop_sizes) > 1 or len(spacings) > 1:
        raise ValueError(
            f"config.json incohérents entre folds : crop_size={crop_sizes}, "
            f"target_spacing={spacings}. Le bundle de soumission doit contenir des "
            f"folds entraînés avec la même résolution d'entrée."
        )

    crop_size = next(iter(crop_sizes)) if crop_sizes else DEFAULT_CROP_SIZE
    target_spacing = next(iter(spacings)) if spacings else DEFAULT_TARGET_SPACING
    return crop_size, target_spacing


def build_model_from_config(cfg: dict):
    """Reconstruit le modèle d'un fold à partir de son config.json, en respectant
    l'architecture réellement utilisée à l'entraînement (train.py y écrit
    "architecture": "ResNet" -- voir --model dans train.py).
    """
    architecture = cfg.get("architecture", "ResNet")

    if architecture == "ResNet":
        return build_model(
            base_channels=cfg["base_channels"],
            depth=cfg["depth"],
            attention=cfg["attention"],
            se_reduction=cfg["se_reduction"],
            cbam_reduction=cfg["cbam_reduction"],
            cbam_spatial_kernel=cfg["cbam_spatial_kernel"],
            mlp_layers=cfg["mlp_layers"], mlp_hidden=cfg["mlp_hidden"], dropout=cfg["dropout"], pool_mode=cfg["pool_mode"],
        )
    raise ValueError(f"architecture inconnue dans config.json : {architecture!r} (attendu:  'ResNet')")


def run_preprocessing(uids, niftis_dir: Path, cache_dir: Path, crop_size, target_spacing, jobs: int = PREPROCESS_JOBS):
    """Prétraite les NIfTI de test correspondant à `uids` vers `cache_dir`.

    Retourne (succès, échecs) : deux listes de uids. Un uid finit dans `échecs` soit
    parce que son .nii.gz est introuvable, soit parce que process_single_file a
    rencontré une erreur (masque vide, image aberrante, etc.). Aucune de ces
    informations n'est loggée ici (voir preprocess_files(quiet=True) ci-dessous) —
    seul le décompte global est utilisé en interne pour router vers la prédiction de
    repli.
    """
    files = [
        niftis_dir / f"{uid}.nii.gz"
        for uid in uids
        if (niftis_dir / f"{uid}.nii.gz").exists()
    ]

    results = preprocess_files(
        files,
        cache_dir,
        crop_size=crop_size,
        target_spacing=target_spacing,
        jobs=jobs,
        show_progress=False,  # cf. docstring de preprocess_files : évite de saturer le log dans le container
        quiet=True,           # aucun nom de fichier de test dans les logs
    )

    succeeded = [uid for uid in uids if results.get(uid, False)]
    failed = [uid for uid in uids if not results.get(uid, False)]
    return succeeded, failed


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sub_format = pd.read_csv(SUBMISSION_FORMAT_CSV)
    uids = [str(u) for u in sub_format["uid"].tolist()]

    default_prob = get_default_probability()
    uid_to_preds = {uid: [] for uid in uids}

    fold_dirs = sorted(RESULTS_DIR.glob("fold_*"))
    if not fold_dirs:
        # Erreur sur le bundle du modèle lui-même (pas sur les données de test) :
        # légitime de la laisser remonter.
        raise FileNotFoundError(f"Aucun sous-dossier 'fold_*' dans {RESULTS_DIR}")

    fold_configs = []
    for fold_dir in fold_dirs:
        with open(fold_dir / "config.json") as f:
            fold_configs.append(json.load(f))

    # crop_size/target_spacing proviennent des config.json des folds (écrits par
    # train.py), pas d'une constante dupliquée ici -- voir resolve_preprocessing_params.
    crop_size, target_spacing = resolve_preprocessing_params(fold_configs)

    # Cache local temporaire pour les .npy du test set : pas de chemin en dur
    # partagé avec preprocess_dataset.py, nettoyage automatique en sortie de bloc.
    with tempfile.TemporaryDirectory(prefix="test_cache_", dir=".") as tmp_dir:
        cache_dir = Path(tmp_dir)
        succeeded_uids, failed_uids = run_preprocessing(
            uids, NIFTIS_DIR, cache_dir, crop_size=crop_size, target_spacing=target_spacing
        )

        if succeeded_uids:
            test_ds = InferenceDataset(succeeded_uids, cache_dir=cache_dir)
            test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_skip_none)

            models_ensemble = []
            for fold_dir, cfg in zip(fold_dirs, fold_configs):
                model = build_model_from_config(cfg)
                model.load_state_dict(torch.load(fold_dir / "model.pt", map_location=device))
                model.to(device).eval()

                models_ensemble.append({"model": model, "temperature": cfg["temperature"]})

            with torch.no_grad():
                for batch in test_loader:
                    if batch is None:
                        continue
                    x, batch_uids = batch
                    x = x.to(device)
                    for member in models_ensemble:
                        logits = member["model"](x).cpu().numpy()
                        probs = probs_from_logits(logits, temperature=member["temperature"])
                        for uid, prob in zip(batch_uids, probs):
                            uid_to_preds[str(uid)].append(prob)

    final_uids, final_probs = [], []
    for uid in uids:
        preds = uid_to_preds.get(uid, [])
        final_uids.append(uid)
        final_probs.append(float(np.mean(preds)) if preds else default_prob)

    submission_df = pd.DataFrame({"uid": final_uids, "is_pathologic": final_probs})
    submission_df.to_csv(OUTPUT_CSV, index=False)
    logger.success(f"Soumission générée avec succès -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
