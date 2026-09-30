import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
import torch
from torch.utils.data import Dataset
import monai.transforms as mt

from .preprocess_dataset import (
    DEFAULT_CROP_SIZE,
    DEFAULT_TARGET_SPACING,
)

@dataclass
class AugmentConfig:
    p_noise: float = 1.0
    noise_std: float = 0.2
    p_affine: float = 1.0
    max_rotate_deg: float = 10.0
    max_translate_vox: float = 3.0 
    max_scale_delta: float = 0.20
    p_elastic: float = 1.0
    elastic_alpha: float = 5.0
    elastic_grid_spacing: float = 15.0
    p_shift_intensity: float = 1.0
    intensity_shift_delta: float = 0.15
    p_blur: float = 1.0
    blur_sigma_max: float = 0.8
    p_dropout: float = 1.0
    dropout_frac: float = 0.1
    dropout_holes: int = 3
    p_blobs: float = 1.0
    n_blobs_min: int = 25
    n_blobs_max: int = 35
    blob_radius_min: float = 0.5
    blob_radius_max: float = 4.0
    blob_intensity_min: float = 0.95
    blob_intensity_max: float = 1.0
    p_flip: float = 0.5

def add_spherical_blobs(
    arr: np.ndarray,
    rng: np.random.Generator,
    n_blobs_min: int = 25,
    n_blobs_max: int = 35,
    radius_min: float = 0.5,
    radius_max: float = 4.0,
    intensity_min: float = 0.95,
    intensity_max: float = 1.0,
) -> np.ndarray:
    """Ajoute ~30 'grumeaux' sphériques (positions tirées uniformément dans le
    volume, rayons entre `radius_min` et `radius_max` voxels, intensité quasi
    égale au max de l'image) afin de simuler des points chauds / artefacts de
    reconstruction et de rendre le modèle plus robuste à ce type de bruit
    (potentiellement plus fréquent sur certains sites/scanners).

    Opère sur un volume (Z, Y, X) en numpy pur, AVANT le passage en tenseur :
    les grumeaux subissent ainsi les mêmes transformations géométriques
    (rotation, élastique, flip...) que le reste du volume dans le pipeline
    MONAI qui suit, ce qui évite des artefacts "collés" toujours parfaitement
    sphériques et alignés aux axes après augmentation.
    """
    out = arr.copy()
    shape = np.asarray(out.shape, dtype=np.float64)  # (Z, Y, X)

    max_val = float(out.max())
    if max_val <= 0:
        return out

    n_blobs = int(rng.integers(n_blobs_min, n_blobs_max + 1))

    for _ in range(n_blobs):
        # Centre tiré uniformément dans le volume (coordonnées continues,
        # donc pas de biais vers les voxels entiers).
        center = rng.uniform(0.0, 1.0, size=3) * (shape - 1)
        radius = rng.uniform(radius_min, radius_max)
        intensity = rng.uniform(intensity_min, intensity_max) * max_val

        lo = np.maximum(0, np.floor(center - radius)).astype(int)
        hi = np.minimum(shape - 1, np.ceil(center + radius)).astype(int)

        zi, yi, xi = np.meshgrid(
            np.arange(lo[0], hi[0] + 1),
            np.arange(lo[1], hi[1] + 1),
            np.arange(lo[2], hi[2] + 1),
            indexing="ij",
        )
        dist2 = (zi - center[0]) ** 2 + (yi - center[1]) ** 2 + (xi - center[2]) ** 2
        mask = dist2 <= radius ** 2

        box = out[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
        # np.maximum : on éclaircit seulement (jamais d'assombrissement d'un
        # voxel déjà plus intense que le grumeau tiré).
        box[mask] = np.maximum(box[mask], intensity)
        out[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = box

    return out


def random_augment(arr: np.ndarray, rng: np.random.Generator, cfg: Optional[AugmentConfig] = None) -> np.ndarray:
    """Applique le stack d'augmentations MONAI à l'intérieur de la petite boîte."""
    cfg = cfg or AugmentConfig()

    if rng.random() < cfg.p_blobs:
        arr = add_spherical_blobs(
            arr,
            rng,
            n_blobs_min=cfg.n_blobs_min,
            n_blobs_max=cfg.n_blobs_max,
            radius_min=cfg.blob_radius_min,
            radius_max=cfg.blob_radius_max,
            intensity_min=cfg.blob_intensity_min,
            intensity_max=cfg.blob_intensity_max,
        )

    max_rot_rad = np.deg2rad(cfg.max_rotate_deg)
    
    # Format (C, D, H, W) attendu par MONAI
    tensor = torch.from_numpy(arr).unsqueeze(0).float()

    transforms = [
        mt.RandGaussianNoise(prob=cfg.p_noise, mean=0.0, std=cfg.noise_std),
        mt.RandAffine(
            prob=cfg.p_affine,
            rotate_range=(max_rot_rad, max_rot_rad, max_rot_rad),
            translate_range=(cfg.max_translate_vox, cfg.max_translate_vox, cfg.max_translate_vox),
            scale_range=(cfg.max_scale_delta, cfg.max_scale_delta, cfg.max_scale_delta),
            mode="bilinear",
            padding_mode="reflection",
        ),
        mt.Rand3DElastic(
            prob=cfg.p_elastic,
            sigma_range=(cfg.elastic_grid_spacing, cfg.elastic_grid_spacing + 1.5),
            magnitude_range=(cfg.elastic_alpha, cfg.elastic_alpha + 0.5),
            mode="bilinear",
            padding_mode="reflection",
        ),
        mt.RandShiftIntensity(prob=cfg.p_shift_intensity, offsets=cfg.intensity_shift_delta),
        mt.RandGaussianSmooth(
            prob=cfg.p_blur, 
            sigma_x=(0.0, cfg.blur_sigma_max),
            sigma_y=(0.0, cfg.blur_sigma_max),
            sigma_z=(0.0, cfg.blur_sigma_max)
        ),
        mt.RandCoarseDropout(
            prob=cfg.p_dropout,
            holes=cfg.dropout_holes,
            spatial_size=int(round(cfg.dropout_frac * min(arr.shape))),
            fill_value=0.0
        ),
        mt.RandFlip(
            prob=cfg.p_flip,    # Probabilité d'appliquer le retournement (ex: 0.5)
            spatial_axis=2       # Inversion Gauche/Droite pour un format (C, D, H, W)
        )
    ]

    transform_pipeline = mt.Compose(transforms)
    
    # Forcer la reproductibilité par échantillon/époque à partir du générateur global
    seed = int(rng.integers(0, 2**31 - 1))
    transform_pipeline.set_random_state(seed=seed)
    torch.manual_seed(seed)

    augmented_tensor = transform_pipeline(tensor)
    augmented_arr = augmented_tensor.squeeze(0).numpy()
    return np.clip(augmented_arr, 0.0, None).astype(np.float32)


def worker_init_fn(worker_id: int) -> None:
    """
    À passer en `worker_init_fn=` du DataLoader dès que num_workers > 0.

    Problème corrigé : `.rng` (utilisé par DaTscanDataset ET DaTscanOnlineDataset,
    voir les deux classes ci-dessous) n'était initialisé qu'une seule fois, dans le
    process parent (`__init__`). Sans `persistent_workers=True`, le Dataset est
    re-forké/re-picklé à CHAQUE époque pour peupler les workers : chacun repart donc
    du même état de générateur (l'objet numpy Generator sérialisé), et cet état
    n'est jamais renvoyé ni persisté côté parent. Concrètement, la séquence
    d'augmentations (et, pour DaTscanOnlineDataset, la séquence de dégradations de
    résolution) tend à se répéter d'une époque à l'autre pour les échantillons
    traités par un même worker, ce qui réduit silencieusement l'effet régularisant
    de tout le pipeline d'augmentation.

    Fix : PyTorch attribue déjà, en interne, une seed différente à chaque worker ET
    à chaque nouvelle époque (torch.initial_seed() == seed_de_l'époque + worker_id).
    On s'en sert pour recréer, dans le worker lui-même, un rng numpy propre à ce
    worker et à cette époque.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return
    seed = torch.initial_seed() % (2**32 - 1)
    worker_info.dataset.rng = np.random.default_rng(seed)


class DaTscanDataset(Dataset):
    """[Pipeline historique] Charge les volumes 3D déjà prétraités (.npy, produits
    par preprocess_dataset.py en amont) depuis `cache_dir`, et applique la fonction
    random_augment.

    Important : si num_workers > 0 dans le DataLoader, passer
    `worker_init_fn=worker_init_fn` (défini ci-dessus), sans quoi tous les workers
    partagent le même état d'augmentation d'une époque à l'autre (voir docstring
    de `worker_init_fn`).
    """
    def __init__(self, df: pd.DataFrame, cache_dir: Path, augment_cfg: Optional[AugmentConfig] = None, is_train: bool = False, seed: int = 42):
        self.df = df.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.augment_cfg = augment_cfg
        self.is_train = is_train
        # Générateur indépendant pour le pipeline d'augmentation.
        # Remplacé par worker_init_fn dans chaque worker lorsque num_workers > 0.
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = str(row["uid"])
        
        npy_path = self.cache_dir / f"{uid}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Fichier manquant dans le cache : {npy_path}")
            
        arr = np.load(npy_path)
        
        if self.is_train and self.augment_cfg is not None:
            arr = random_augment(arr, self.rng, self.augment_cfg)
            
        tensor = torch.from_numpy(arr).float().unsqueeze(0) # Shape finale: (1, Z, Y, X)
        
        if "is_pathologic" in row:
            label = torch.tensor(row["is_pathologic"], dtype=torch.float32)
            return tensor, label, uid
        
        return tensor, uid


def get_stratified_splits(labels_csv: Path, n_splits: int = 5, seed: int = 42):
    """Génère des index de splits stratifiés par Site et par Label.

    Note : la colonne 'site' n'existe pas dans le train_labels.csv fourni par
    l'organisateur (voir context.txt) — c'est une colonne construite à partir de la
    résolution/taille des images d'entrée, utilisée comme proxy du centre
    d'acquisition pour stratifier la CV et se prémunir d'un split qui
    surreprésenterait un scanner/protocole donné dans un seul fold.
    """
    df = pd.read_csv(labels_csv)
    if "site" not in df.columns:
        raise ValueError("Le fichier CSV doit contenir une colonne 'site' pour la stratification des centres.")
        
    df["strat_key"] = df["site"].astype(str) + "_" + df["is_pathologic"].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(df, df["strat_key"]))
    return df, splits
    
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

