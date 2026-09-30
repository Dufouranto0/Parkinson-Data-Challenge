import argparse
import json
import gc
import time
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from module.dataloader import (
    DaTscanDataset,
    get_stratified_splits,
    get_stratified_group_splits,
    AugmentConfig,
    worker_init_fn,
)
from module.model import build_model, collect_logits, evaluate_metrics, fit_temperature_robust, FocalLoss
from module.preprocess_dataset import (
    DEFAULT_CROP_SIZE,
    DEFAULT_TARGET_SPACING,
)

# Reprise sur crash / arrêt machine (voir _load_completed_fold ci-dessous) : clés
# d'arguments ignorées lors de la comparaison entre le run courant et un fold déjà
# présent sur disque. Elles ne changent jamais le résultat de l'entraînement
# (mêmes poids, mêmes métriques de validation) -- seulement la vitesse ou
# l'emplacement disque -- donc un fold entraîné avec --num_workers 8 reste valable
# pour reprendre un run relancé avec --num_workers 20, tant que results_dir pointe
# vers le même dossier qu'on est justement en train d'inspecter.
RESUME_IGNORED_ARGS = {"num_workers", "pin_memory", "results_dir"}


def _resume_mismatch_reasons(saved_args: dict, current_args: dict) -> list:
    """Liste lisible des hyperparamètres qui diffèrent entre un fold déjà présent
    sur disque (`saved_args`, lu depuis son config.json) et le run courant
    (`current_args`, vars(args)) -- ignore RESUME_IGNORED_ARGS. Liste vide =
    hyperparamètres identiques = fold réutilisable tel quel."""
    reasons = []
    for key in sorted(set(saved_args) | set(current_args)):
        if key in RESUME_IGNORED_ARGS:
            continue
        saved_val = saved_args.get(key, "<absent>")
        current_val = current_args.get(key, "<absent>")
        if saved_val != current_val:
            reasons.append(f"--{key} : sauvegardé={saved_val!r} / actuel={current_val!r}")
    return reasons


def _load_completed_fold(fold_dir: Path, args) -> Optional[dict]:
    """Reprise sur crash : si `fold_dir` contient déjà un `model.pt` ET un
    `config.json` complets, produits avec des hyperparamètres identiques (à
    RESUME_IGNORED_ARGS près) à `args`, retourne ce config.json déjà chargé -- le
    fold correspondant sera alors sauté par run_cross_validation, qui passera
    directement au split suivant. Retourne None sinon (fold absent, incomplet, ou
    entraîné avec d'autres hyperparamètres) -> ce fold est (ré)entraîné depuis le
    début, comme avant l'ajout de ce mécanisme.

    Ne reprend JAMAIS un fold interrompu EN COURS d'entraînement (pas de
    sauvegarde par époque dans ce script) : un fold coupé avant l'écriture finale
    de model.pt/config.json (typiquement le fold en cours au moment d'un arrêt
    machine) est intégralement ré-entraîné depuis l'époque 1, comme n'importe quel
    fold jamais commencé.
    """
    model_path = fold_dir / "model.pt"
    config_path = fold_dir / "config.json"
    if not (model_path.exists() and config_path.exists()):
        return None

    try:
        with open(config_path) as f:
            saved_config = json.load(f)
    except Exception as e:
        print(f"[Reprise] {config_path} illisible ({e}) -> fold ré-entraîné depuis le début.")
        return None

    saved_args = saved_config.get("train_args")
    if saved_args is None:
        print(f"[Reprise] {config_path} ne contient pas 'train_args' (généré par une version antérieure de train.py) -> fold ré-entraîné depuis le début.")
        return None

    reasons = _resume_mismatch_reasons(saved_args, vars(args))
    if reasons:
        print(f"[Reprise] {fold_dir.name} existe déjà mais avec des hyperparamètres différents -> fold ré-entraîné depuis le début. Différences :")
        for r in reasons:
            print(f"    {r}")
        return None

    return saved_config


def build_train_val_datasets(args, train_df, val_df, augment_cfg, fold_seed):
    """Construit les datasets train/val
    """

    train_ds = DaTscanDataset(train_df, cache_dir=args.cache_dir, augment_cfg=augment_cfg, is_train=True, seed=fold_seed)
    val_ds = DaTscanDataset(val_df, cache_dir=args.cache_dir, is_train=False)

    return train_ds, val_ds


def run_cross_validation(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Début de l'entraînement Multi-Split sur l'appareil : {device}")
    print(f"Configuration matérielle : num_workers={args.num_workers} | pin_memory={args.pin_memory}")

    pipeline_name = "cached (.npy précalculés)"
    print(f"Pipeline de données : {pipeline_name}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    augment_cfg = AugmentConfig(
        p_noise=args.p_noise, noise_std=args.noise_std,
        p_affine=args.p_affine, max_rotate_deg=args.max_rotate_deg, max_translate_vox=args.max_translate_vox, max_scale_delta=args.max_scale_delta,
        p_elastic=args.p_elastic, elastic_alpha=args.elastic_alpha, elastic_grid_spacing=args.elastic_grid_spacing,
        p_shift_intensity=args.p_shift_intensity, intensity_shift_delta=args.intensity_shift_delta,
        p_blur=args.p_blur, blur_sigma_max=args.blur_sigma_max,
        p_dropout=args.p_dropout, dropout_frac=args.dropout_frac, dropout_holes=args.dropout_holes,
        p_blobs=args.p_blobs, n_blobs_min=args.n_blobs_min, n_blobs_max=args.n_blobs_max,
        blob_radius_min=args.blob_radius_min, blob_radius_max=args.blob_radius_max,
        blob_intensity_min=args.blob_intensity_min, blob_intensity_max=args.blob_intensity_max,
    )
    
    df, splits = get_stratified_group_splits(args.train_labels, n_splits=args.n_splits, seed=args.seed)
    cv_summary = []

    # Sauvegarde de la prévalence globale du train, utilisée par main.py comme
    # prédiction de repli si le prétraitement ou le chargement d'un examen de test
    # échoue. Important : dans le container de soumission, data/ ne contient QUE
    # submission_format.csv et les NIfTI de test — aucun fichier de labels (train ou
    # test) n'y est présent. Cette valeur doit donc être précalculée ici, à
    # l'entraînement, et embarquée dans results/ avec les poids du modèle.
    fallback_probability = float(df["is_pathologic"].mean())
    with open(results_dir / "fallback_probability.json", "w") as f:
        json.dump({"fallback_probability": fallback_probability}, f, indent=2)
    print(f"Prévalence globale du train sauvegardée comme repli d'inférence : {fallback_probability:.4f} -> {results_dir / 'fallback_probability.json'}")

    # Fonction de perte utilisée pour la DESCENTE DE GRADIENT uniquement. L'early
    # stopping et la sélection du meilleur checkpoint (best_val_loss, plus bas)
    # restent TOUJOURS basés sur la val log loss (sklearn.metrics.log_loss via
    # evaluate_metrics), quel que soit ce choix — ce sont des objets complètement
    # indépendants dans la boucle ci-dessous.
    #
    # Le scheduler LR, lui, diffère désormais selon l'architecture (cf. plus bas,
    # au moment de sa construction) : ReduceLROnPlateau piloté par la val log loss
    # pour ResNet, warmup linéaire + cosine decay mis à jour à chaque batch
    # pour ConvNeXt (recette standard pour cette famille d'architectures).
    if args.loss == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"Loss d'entraînement : Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
        print("Loss d'entraînement : BCEWithLogitsLoss")
    print("Sélection du meilleur checkpoint : toujours basée sur la Val LogLoss (indépendante de la loss d'entraînement ci-dessus).")
    
    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        fold_dir = results_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Reprise sur crash / arrêt machine : si ce fold a déjà été entraîné
        # jusqu'au bout avec des hyperparamètres identiques (cf.
        # _load_completed_fold), on saute directement au split suivant plutôt que
        # de tout reprendre depuis le fold 1. Rien d'autre n'est fait pour ce fold
        # (pas de DataLoader, pas de modèle construit) : on va droit au fold
        # suivant.
        completed_config = _load_completed_fold(fold_dir, args)
        if completed_config is not None:
            print(f"\n" + "="*60)
            print(
                f"[Reprise] SPLIT {fold}/{args.n_splits} déjà entraîné (trouvé "
                f"{fold_dir}/model.pt + config.json cohérent avec les paramètres "
                f"actuels) -> passage au split suivant sans réentraîner."
            )
            print(
                f"    Val LogLoss (sauvegardée) : {completed_config['val_log_loss']:.4f} | "
                f"AUROC : {completed_config['val_auroc']:.4f} | T : {completed_config['temperature']:.4f}"
            )
            print(f"="*60)
            cv_summary.append(completed_config["val_log_loss"])
            continue

        print(f"\n" + "="*60)
        print(f"SPLIT {fold}/{args.n_splits} | Initialisation des DataLoaders...")
        print(f"="*60)

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        print(f"Taille du sous-ensemble d'ENTRAÎNEMENT (Train) : {len(train_df)} sujets")
        print(f"Taille du sous-ensemble de VALIDATION (Val)     : {len(val_df)} sujets")
        
        train_ds, val_ds = build_train_val_datasets(args, train_df, val_df, augment_cfg, fold_seed=args.seed + fold)
        
        # workers configurables via CLI pour accélérer le traitement MONAI
        train_loader = DataLoader(
            train_ds, 
            batch_size=args.batch_size, 
            shuffle=True, 
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            worker_init_fn=worker_init_fn,
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory
        )
        

        if args.model == "ResNet":
            model = build_model(
                base_channels=args.base_channels,
                depth=args.depth,
                attention=args.attention,
                se_reduction=args.se_reduction,
                cbam_reduction=args.cbam_reduction,
                cbam_spatial_kernel =args.cbam_spatial_kernel,
                mlp_layers=args.mlp_layers,
                mlp_hidden=args.mlp_hidden,
                dropout=args.dropout,
                pool_mode=args.pool_mode,
            ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Scheduler LR : ReduceLROnPlateau piloté par la Val LogLoss (cf. commentaire
        # plus haut sur le choix du scheduler selon l'architecture). Réduit le LR
        # d'un facteur `args.lr_factor` dès que la Val LogLoss stagne (pas
        # d'amélioration > `args.lr_threshold`) pendant `args.lr_patience` époques
        # consécutives. Totalement indépendant de la patience d'early stopping
        # (`args.patience`), qui a son propre compteur plus bas dans la boucle.
        scheduler_mode = "per_epoch_val_loss"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            threshold=args.lr_threshold,
            min_lr=args.lr_min,
        )
        print(
            f"Scheduler LR : ReduceLROnPlateau (mode=min sur Val LogLoss, "
            f"factor={args.lr_factor}, patience={args.lr_patience} époques, "
            f"threshold={args.lr_threshold}, min_lr={args.lr_min})"
        )

        
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            
            # Utilisation de tqdm pour surveiller l'avancement des batches dans l'époque
            model.train()
            total_loss, n = 0.0, 0
            
            pbar = tqdm(
                train_loader, 
                desc=f"Fold {fold}/{args.n_splits} | Époque {epoch:03d}/{args.epochs:03d}", 
                leave=False
            )
            
            for x, y, _ in pbar:
                x, y = x.to(device), y.to(device)
                y_target = y * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing if args.label_smoothing > 0 else y
                optimizer.zero_grad()
                loss = criterion(model(x), y_target)
                loss.backward()
                optimizer.step()
                if scheduler_mode == "per_step":
                    scheduler.step()
                
                total_loss += loss.item() * x.size(0)
                n += x.size(0)
                
                # Mise à jour des métriques à la volée sur la barre de progression
                pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
                
            train_loss = total_loss / max(n, 1)
            epoch_duration = time.time() - t0
            samples_per_sec = n / epoch_duration
            
            # Évaluation sur le set de validation
            val_logits, val_labels, _ = collect_logits(model, val_loader, device)
            val_metrics = evaluate_metrics(val_logits, val_labels)
            val_loss = val_metrics["log_loss"]
            lr_before_step = optimizer.param_groups[0]["lr"]
            if scheduler_mode == "per_epoch_val_loss":
                scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr < lr_before_step:
                print(f"[Scheduler] Val LogLoss stagnante depuis {args.lr_patience} époques -> LR réduit : {lr_before_step:.2e} -> {current_lr:.2e}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                marker = "[Nouveau Meilleur]"
            else:
                patience_counter += 1
                marker = f" (Patience: {patience_counter}/{args.patience})"
                
            print(
                f"Époque {epoch:03d}/{args.epochs:03d} - Terminée en {epoch_duration:.1f}s ({samples_per_sec:.1f} scans/s) | "
                f"Train Loss: {train_loss:.4f} | Val LogLoss: {val_loss:.4f} | AUROC: {val_metrics['auroc']:.4f} | LR: {current_lr:.2e}{marker}"
            )
            
            if patience_counter >= args.patience:
                print(f"Arrêt précoce déclenché au Fold {fold} à l'époque {epoch}.")
                break
                
        # Phase finale de calibration
        model.load_state_dict(best_state)
        val_logits, val_labels, _ = collect_logits(model.to(device), val_loader, device)
        
        print(f"Calcul de la température de calibration (Bootstrap) pour le Fold {fold}...")
        temperature = fit_temperature_robust(val_logits, val_labels, n_bootstrap=50, seed=args.seed)
        final_metrics = evaluate_metrics(val_logits, val_labels, temperature=temperature)
        
        print(f"Résultat Fold {fold} -> T: {temperature:.4f} | LogLoss Calibrée: {final_metrics['log_loss']:.4f} | AUROC: {final_metrics['auroc']:.4f}")
        
        torch.save(best_state, fold_dir / "model.pt")
        config = {
            # Architecture -- nécessaire à build_model() en inférence
            # (main.py) pour reconstruire le bon réseau par fold.
            "architecture": args.model,  # "ResNet"
            "base_channels": args.base_channels,
            "mlp_layers": args.mlp_layers, "mlp_hidden": args.mlp_hidden, "dropout": args.dropout, "pool_mode": args.pool_mode,
            # Calibration + métriques de validation
            "temperature": temperature, "val_log_loss": final_metrics["log_loss"], "val_auroc": final_metrics["auroc"],
            # Résolution d'entrée utilisée pour produire les données d'entraînement
            # relue par main.py à l'inférence plutôt que dupliquée en
            # dur, pour ne jamais désynchroniser train et inférence. main.py ne lit
            # QUE ces deux clés-ci pour son propre prétraitement (jamais les clés
            # "degrade_*", qui ne concernent que l'entraînement).
            "crop_size": list(args.crop_size),
            "target_spacing": list(args.target_spacing),
            # Traçabilité de la pipeline de données utilisée pour ce run (purement
            # informatif, non consommé par main.py).
            "data_pipeline": "cached",
            # Trace complète de tous les hyperparamètres CLI de ce run, pour
            # reproductibilité/audit (redondant avec les clés ci-dessus par endroits,
            # c'est voulu : les clés du dessus sont le contrat minimal consommé par
            # le code, celle-ci est l'archive complète).
            "train_args": vars(args),
        }
        if args.model == "ResNet":

            config["depth"] = args.depth
            config["attention"] = args.attention
            config["se_reduction"] = args.se_reduction
            config["cbam_reduction"] = args.cbam_reduction
            config["cbam_spatial_kernel"] = args.cbam_spatial_kernel

        with open(fold_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
            
        cv_summary.append(final_metrics["log_loss"])
        del model, optimizer, scheduler, best_state
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n " + "="*40)
    print(f"FIN DE LA CROSS-VALIDATION MULTI-SPLIT")
    print(f"Moyenne LogLoss CV : {np.mean(cv_summary):.4f}")
    print(f"="*40)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_labels", type=str, required=True)

    # Pipeline de données : fournir EXACTEMENT UN des deux.
    p.add_argument(
        "--cache_dir", type=str, default=None,
        help="[Pipeline historique] Dossier contenant les .npy déjà précalculés par "
             "preprocess_dataset.py (CLI).",
    )
    
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--label_smoothing", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=42)

    # Scheduler LR (ReduceLROnPlateau, piloté par la Val LogLoss -- indépendant de
    # --patience ci-dessus, qui régit l'early stopping global).
    p.add_argument("--lr_patience", type=int, default=15, help="Nombre d'époques de stagnation de la Val LogLoss avant réduction du LR.")
    p.add_argument("--lr_factor", type=float, default=0.5, help="Facteur multiplicatif appliqué au LR lors d'une réduction (nouveau_lr = lr * factor).")
    p.add_argument("--lr_threshold", type=float, default=1e-3, help="Amélioration minimale de la Val LogLoss pour ne pas compter comme stagnation.")
    p.add_argument("--lr_min", type=float, default=1e-6, help="LR plancher, jamais franchi par le scheduler.")

    # Fonction de perte pour l'entraînement (la sélection du meilleur checkpoint
    # reste toujours basée sur la val log loss, cf. run_cross_validation)
    p.add_argument("--loss", type=str, default="focal", choices=["bce", "focal"], help="Fonction de perte utilisée pour la descente de gradient.")
    p.add_argument("--focal_alpha", type=float, default=0.25, help="Poids alpha de la Focal Loss (rééquilibrage de PRÉVALENCE entre classes).")
    p.add_argument("--focal_gamma", type=float, default=2, help="Exposant gamma de la Focal Loss (atténue la contribution des exemples faciles/bien classés).")

    # Résolution 
    p.add_argument("--crop_size", type=int, nargs=3, default=list(DEFAULT_CROP_SIZE), help="Taille du crop (voxels).")
    p.add_argument("--target_spacing", type=float, nargs=3, default=list(DEFAULT_TARGET_SPACING), help="Spacing isotrope (mm) cible du rééchantillonnage final.")

    # Paramètres d'accélération matérielle
    p.add_argument("--num_workers", type=int, default=16, help="Nombre de processus CPU parallèles pour le chargement/MONAI")
    p.add_argument("--no_pin_memory", dest="pin_memory", action="store_false", help="Désactive le chargement asynchrone VRAM")
    p.set_defaults(pin_memory=True)
    
    p.add_argument("--model", type=str, default="ResNet", choices=["ResNet"])

    # ResNet
    p.add_argument("--base_channels", type=int, default=16)
    p.add_argument("--mlp_layers", type=int, default=1)
    p.add_argument("--mlp_hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--pool_mode", type=str, default="avg_max")
    p.add_argument("--depth", type=str, default="resnet18", choices=["resnet10", "resnet18"])
    p.add_argument("--attention", type=str, default="cbam", choices=["none", "se", "cbam"])
    p.add_argument("--se_reduction", type=int, default=8)
    p.add_argument("--cbam_reduction", type=int, default=8)
    p.add_argument("--cbam_spatial_kernel", type=int, default=7)
    
    p.add_argument("--p_noise", type=float, default=0.9)
    p.add_argument("--noise_std", type=float, default=0.10)
    p.add_argument("--p_affine", type=float, default=0.9)
    p.add_argument("--max_rotate_deg", type=float, default=40)
    p.add_argument("--max_translate_vox", type=float, default=8.0)
    p.add_argument("--max_scale_delta", type=float, default=0.1)
    p.add_argument("--p_elastic", type=float, default=0.3)
    p.add_argument("--elastic_alpha", type=float, default=4.5)
    p.add_argument("--elastic_grid_spacing", type=float, default=6)
    p.add_argument("--p_shift_intensity", type=float, default=0.09)
    p.add_argument("--intensity_shift_delta", type=float, default=0.17)
    p.add_argument("--p_blur", type=float, default=0.2)
    p.add_argument("--blur_sigma_max", type=float, default=0.63)
    p.add_argument("--p_dropout", type=float, default=0.6)
    p.add_argument("--dropout_frac", type=float, default=0.1)
    p.add_argument("--dropout_holes", type=int, default=3)
    p.add_argument("--p_blobs", type=float, default=0.15, help="Probabilité d'ajouter les grumeaux sphériques.")
    p.add_argument("--n_blobs_min", type=int, default=20)
    p.add_argument("--n_blobs_max", type=int, default=35)
    p.add_argument("--blob_radius_min", type=float, default=0.5)
    p.add_argument("--blob_radius_max", type=float, default=4.0)
    p.add_argument("--blob_intensity_min", type=float, default=0.95)
    p.add_argument("--blob_intensity_max", type=float, default=1.0)
    
    args = p.parse_args()
    run_cross_validation(args)
