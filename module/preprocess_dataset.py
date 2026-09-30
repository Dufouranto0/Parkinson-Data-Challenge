#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from loguru import logger
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

DEFAULT_CROP_SIZE = (90, 90, 90)
DEFAULT_TARGET_SPACING = (1.5, 1.5, 1.5)


def process_single_file(img_path: Path, out_dir: Path, crop_size: tuple, quiet: bool = False, target_spacing: tuple = DEFAULT_TARGET_SPACING):
    """
    Pipeline de prétraitement rapide avec atténuation gaussienne successive.

    `quiet=True` supprime le log d'erreur par fichier (qui contient le nom du
    fichier, donc l'uid) : à utiliser quand cette fonction tourne sur des données de
    test, où logger l'identité d'un examen est interdit par le règlement de la
    compétition. Le CLI (traitement du train, en local) garde quiet=False.
    """
    try:
        # 1. Chargement et rééchantillonnage isotropique
        img = sitk.ReadImage(str(img_path), sitk.sitkFloat32)
        img = sitk.DICOMOrient(img, 'LPS')
        original_spacing = img.GetSpacing()
        original_size = img.GetSize()

        # Taille physique cible en mm : 80 * 1.5 = 120mm
        taille_physique_mm = [c * t for c, t in zip(crop_size, target_spacing)]
        #taille_physique_mm[0] = round(DEFAULT_CROP_SIZE[0]*3/4)+1 * DEFAULT_TARGET_SPACING[0]

        # Taille du crop en voxels natifs (dépend du patient)
        native_crop_size = [
            int(round(taille_physique_mm[i] / original_spacing[i]))
            for i in range(3)
        ]

        # Extraction de la matrice numpy (Shape: Z, Y, X)
        data = sitk.GetArrayFromImage(img)

        # Cas A : L'image est totalement inversée (négative) -> on la passe en positif
        if data.min() < 0 and data.max() <= 0:
            data = np.abs(data)
        
        # Cas B : L'image bave dans le négatif ou est centrée sur 0 -> on shifte le minimum à 0
        elif data.min() < 0:
            data = data - data.min()
        
        # 2. CALCUL DU CENTRE DE MASSE INITIAL (Forme globale de la tête)
        non_zero = data[data > 0]
        if len(non_zero) == 0:
            raise ValueError("L'image ne contient que des valeurs nulles.")
            
        # =====================================================================
        # CALCUL ADAPTATIF DU TOP-K POUR LE CERVEAU GLOBAL (ROUGH MASK)
        # =====================================================================
        # 1. Définition du volume physique maximal d'un gros cerveau humain + méninges + crâne (~1650 cm³)
        # On prend une marge haute pour être certain d'englober toute la boîte si elle est petite.
        volume_cerveau_max_mm3 = 1_650_000.0  

        # 2. Calcul du volume d'un voxel natif pour ce patient
        voxel_volume_mm3 = original_spacing[0] * original_spacing[1] * original_spacing[2]

        # 3. Nombre théorique de voxels nécessaires pour contenir ce gros cerveau
        k_voxels_cerveau = int(round(volume_cerveau_max_mm3 / voxel_volume_mm3))

        # 4. Extraction adaptative du seuil (Top-K)
        non_zero = data[data > 0]
        if len(non_zero) == 0:
            raise ValueError("L'image ne contient que des valeurs nulles.")

        # SI l'image entière (ou ses voxels non nuls) est plus petite que notre cerveau théorique maximal,
        # cela signifie que l'image est déjà très bien recadrée. On prend alors le seuil minimal (percentile bas)
        # pour ne rien amputer de l'anatomie présente.
        if len(non_zero) <= k_voxels_cerveau:
            # On prend un percentile très bas (ex: 5%) juste pour éliminer le bruit de fond absolu
            rough_threshold = np.percentile(non_zero, 5)
        else:
            # Sinon (image géante avec beaucoup de vide), on isole strictement les K voxels les plus intenses
            rough_threshold = np.partition(non_zero, -k_voxels_cerveau)[-k_voxels_cerveau]

        # 5. Création du masque grossier stable
        rough_mask = (data >= rough_threshold).astype(np.uint8)
        
        # =====================================================================
        # CALCUL DU CENTRE DE MASSE GLOBAL INITIAL VIA SIMPLEITK
        # =====================================================================
        sitk_rough_mask = sitk.GetImageFromArray(rough_mask)
        sitk_rough_mask.CopyInformation(img)
        
        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        shape_stats.Execute(sitk.Cast(sitk_rough_mask, sitk.sitkUInt8))
        
        if not shape_stats.HasLabel(1):
            raise ValueError("Impossible de calculer le centre de masse global.")
            
        rough_com_physical = shape_stats.GetCentroid(1)
        current_com_idx = img.TransformPhysicalPointToIndex(rough_com_physical) # (X, Y, Z)

        # Génération unique des grilles d'indices (Mémoire partagée pour la vitesse)
        z_indices, y_indices, x_indices = np.indices(data.shape)

        # Calcul de la demi-largeur physique max autorisée sur l'axe X (Droite-Gauche)
        taille_x_physique = 60 * target_spacing[0]
        rayon_x_max_mm = taille_x_physique / 2.0

        # Fonction vectorisée ultra-rapide pour appliquer l'atténuation
        def apply_gaussian_weight(input_data, center_idx, sigma_mm):
            # 1. Calcul des distances physiques en mm
            dz = (z_indices - center_idx[2]) * original_spacing[2]
            dy = (y_indices - center_idx[1]) * original_spacing[1]
            dx = (x_indices - center_idx[0]) * original_spacing[0]
            
            # 2. Calcul du poids gaussien standard
            squared_distance = dx**2 + dy**2 + dz**2
            gaussian_weight = np.exp(-squared_distance / (2 * sigma_mm**2))
            
            # 3. Restriction stricte sur l'axe X (Droite-Gauche)
            # Si un voxel est plus éloigné du centre que le rayon_x_max_mm, son poids devient 0
            masque_fenetre_x = (np.abs(dx) <= rayon_x_max_mm)
            
            # 4. Application combinée
            return input_data * gaussian_weight * masque_fenetre_x

        # Normalisation globale initiale
        data = np.clip(data / data.max(), 0.0, 1.0).astype(np.float32)

        # =====================================================================
        # CALCUL DYNAMIQUE DU TOP-K SELON LA RÉSOLUTION NATIVE
        # =====================================================================
        # 1. Calcul du volume d'un seul voxel en mm³ (X_spacing * Y_spacing * Z_spacing)
        voxel_volume_mm3 = original_spacing[0] * original_spacing[1] * original_spacing[2]

        # 2. Définition des volumes cibles biologiques (en mm³) pour chaque étape de l'entonnoir
        # Ces valeurs modélisent la décroissance de la "fenêtre anatomique" visée :
        # - Passe 1 : Striatum élargi et tissu environnant (~30 000 mm³)
        # - Passe 2 : Striatum élargi (~12 000 mm³)
        volumes_cibles_mm3 = [30_000.0, 12_000.0, 5_000.0]

        # 3. Génération automatique de la configuration (Sigma_mm, K_voxels_natifs)
        sigmas_passes = [40.0, 30.0, 20.0]
        configurations_passes = []
        
        for sigma, vol_mm3 in zip(sigmas_passes, volumes_cibles_mm3):
            # Le nombre théorique de voxels est : Volume cible / Volume d'un voxel
            # On applique un max(1, ...) pour éviter un Top-0 sur des images extrêmement low-res
            k_voxels_theorique = max(1, int(round(vol_mm3 / voxel_volume_mm3)))
            configurations_passes.append((sigma, k_voxels_theorique))

        # =====================================================================
        # BOUCLE D'ATTÉNUATION EN ENTONNOIR FACTORISÉE
        # =====================================================================
        attenuated = data
        com_history = [rough_com_physical]

        for i, (sigma, k_vox) in enumerate(configurations_passes):
            # 1. Application de l'atténuation physique (distance au COM précédent)
            attenuated = apply_gaussian_weight(data, current_com_idx, sigma_mm=sigma)
            
            # =====================================================================
            # LISSAGE GAUSSIEN ADAPTATIF (Filtrage du bruit infra-striatal)
            # =====================================================================
            # On définit un sigma de lissage physique de 2 mm (pour lisser le bruit)
            sigma_lissage_mm = 2.5
            
            # Conversion du sigma physique en sigma-voxel pour chaque axe de l'image courante
            sigma_voxels = [sigma_lissage_mm / sp for sp in original_spacing]
            
            # Utilisation de SimpleITK pour un filtrage gaussien ultra-rapide et précis
            sitk_attenuated = sitk.GetImageFromArray(attenuated)
            sitk_attenuated.CopyInformation(img)
            
            # Filtre Gaussien récursif (hautement optimisé mathématiquement)
            gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
            gaussian_filter.SetSigma(sigma_voxels)
            sitk_smoothed = gaussian_filter.Execute(sitk_attenuated)
            
            # Récupération de la matrice lissée pour la recherche du Top-K
            smoothed_array = sitk.GetArrayFromImage(sitk_smoothed).astype(np.float32)
            
            # =====================================================================
            # SEUILLAGE TOP-K SUR L'IMAGE LISSÉE
            # =====================================================================
            smoothed_non_zero = smoothed_array[smoothed_array > 0]
            if len(smoothed_non_zero) == 0:
                continue
                
            if len(smoothed_non_zero) <= k_vox:
                threshold = smoothed_non_zero.min()
            else:
                # La recherche des voxels les plus intenses se fait sur l'image lissée,
                # là où le bruit de petite taille a été "écrasé" et étalé.
                threshold = np.partition(smoothed_non_zero, -k_vox)[-k_vox]
            
            # Le masque binaire final capture le cœur lissé du striatum
            mask = (smoothed_array >= threshold).astype(np.uint8)
            
            # =====================================================================
            # CALCUL DU COM VIA SIMPLEITK (Sur le masque débruité)
            # =====================================================================
            sitk_mask = sitk.GetImageFromArray(mask)
            sitk_mask.CopyInformation(img)
            shape_stats.Execute(sitk_mask)
            
            if shape_stats.HasLabel(1):
                com_physical = shape_stats.GetCentroid(1)
                com_history.append(com_physical)
                current_com_idx = img.TransformPhysicalPointToIndex(com_physical)
            elif not quiet:
                print(f"Masque vide pour la passe {i} (Sigma {sigma}mm, K {k_vox} voxels)")

        # Définition sécurisée du COM avant l'ultra-raffinement continu
        final_com_physical = com_history[-1]

        # Recentrage sur l'axe Gauche-Droite (axe physique X — en convention
        # NIfTI/ITK standard, l'axe physique 0 est toujours l'axe Gauche-Droite,
        # quelle que soit la matrice de direction voxel->physique propre à ce scan).
        # Le COM affiné par les 3 passes d'atténuation suit le pic de fixation
        # striatale, qui est PRÉCISÉMENT le signal pathologique recherché : en cas
        # d'asymétrie de fixation (fréquente et cliniquement attendue dans les scans
        # anormaux), ce COM est donc mécaniquement tiré vers l'hémisphère le plus
        # fixant, ce qui décalerait le crop final de façon corrélée à la
        # sévérité/latéralité de la pathologie elle-même — un biais de cadrage à
        # éviter. Le COM "large" (rough_com_physical) provient lui d'un masque
        # grossier sur l'ensemble de la tête, peu sensible à cette asymétrie focale,
        # et reste une bien meilleure approximation du plan sagittal médian. On
        # réancre donc uniquement la coordonnée X (G/D) du COM final sur celle du
        # COM initial, en conservant les coordonnées Y/Z (A/P, S/I) issues du
        # raffinement.
        final_com_physical = (
            rough_com_physical[0],
            final_com_physical[1],
            final_com_physical[2],
        )

        com_index = img.TransformPhysicalPointToIndex(final_com_physical)

        # 5. DÉCOUPE ET PADDING (Entièrement en espace natif)
        img_size = img.GetSize()
        pad_lower = [0, 0, 0]
        pad_upper = [0, 0, 0]
        
        # On vérifie si l'image native est plus petite que le native_crop_size calculé
        for i in range(3):
            if img_size[i] < native_crop_size[i]:
                needed = native_crop_size[i] - img_size[i]
                pad_lower[i] = needed // 2 + 1
                pad_upper[i] = needed // 2 + 1
                
        if sum(pad_lower) + sum(pad_upper) > 0:
            img = sitk.ConstantPad(img, pad_lower, pad_upper, 0.0)
            com_index = img.TransformPhysicalPointToIndex(final_com_physical)
            img_size = img.GetSize()

        start_index = [0, 0, 0]
        for i in range(3):
            # Utilisation stricte de native_crop_size pour centrer le cube natif
            start = com_index[i] - native_crop_size[i] // 2
            if start < 0:
                start = 0
            if start + native_crop_size[i] > img_size[i]:
                start = img_size[i] - native_crop_size[i]
            start_index[i] = int(start)

        # Découpe du volume au format natif adapté au patient
        cropped_img = sitk.RegionOfInterest(img, native_crop_size, start_index)
        
        # RÉÉCHANTILLONNAGE FINAL : C'est ici qu'on standardise vers la cible (ex: 80x80x80)
        cropped_img = sitk.Resample(
            cropped_img, crop_size, sitk.Transform(), sitk.sitkBSpline, 
            cropped_img.GetOrigin(), target_spacing, cropped_img.GetDirection(), 0.0, cropped_img.GetPixelID()
        )

        # Extraction finale pour la normalisation
        final_array = sitk.GetArrayFromImage(cropped_img).astype(np.float32)

        # 6. Normalisation Min-Max (0.0 - 1.0) — CALCULÉE SUR LE CROP FINAL
        # (80x80x80 @ 1.5mm iso, soit 120mm de côté), et non sur l'image d'entrée
        # complète : le FOV brut peut contenir de la fixation extracrânienne
        # (muqueuse nasale, glandes salivaires) ou du bruit de reconstruction dont
        # l'intensité dépasse celle du striatum, ce qui biaiserait les percentiles
        # de normalisation. À 120mm de côté, le crop reste assez large pour englober
        # l'essentiel du cerveau (striatum + tissu de fond environnant), donc
        # normaliser dans cette fenêtre écarte le bruit périphérique tout en
        # conservant le contraste relatif strié / fond.
        cropped_array = sitk.GetArrayFromImage(cropped_img).astype(np.float32)
        crop_non_zero = cropped_array[cropped_array > 0]
        if len(crop_non_zero) == 0:
            raise ValueError("Le crop final ne contient que des valeurs nulles.")

        norm_low = float(np.percentile(crop_non_zero, 1))
        norm_high = float(np.percentile(crop_non_zero, 99.95))
        if norm_high <= norm_low:
            norm_high = norm_low + 1e-6

        final_array = (cropped_array - norm_low) / (norm_high - norm_low)
        final_array = np.clip(final_array, 0.0, 1.0).astype(np.float32)

        # 7. Sauvegarde .npy
        out_filename = img_path.name.replace(".nii.gz", "").replace(".nii", "") + ".npy"
        out_path = out_dir / out_filename
        np.save(str(out_path), final_array)
        return True

    except Exception as e:
        if not quiet:
            logger.error(f"Erreur lors du traitement de {img_path.name} : {str(e)}")
        return False


def _uid_from_path(p: Path) -> str:
    name = p.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return p.stem


def preprocess_files(
    files,
    out_dir: Path,
    crop_size: tuple = DEFAULT_CROP_SIZE,
    target_spacing: tuple = DEFAULT_TARGET_SPACING,
    jobs: int = 4,
    show_progress: bool = True,
    quiet: bool = False,
) -> dict:
    """Prétraite une liste de fichiers NIfTI vers out_dir, en parallèle.

    Retourne {uid: bool} — True si le .npy correspondant a bien été généré, False
    sinon (fichier corrompu, masque vide sur tous les seuillages, image aberrante,
    etc.). Ne laisse jamais une exception individuelle remonter : c'est ce dict qui
    permet à l'appelant (ex: main.py en inférence) de décider quoi faire des échecs
    plutôt que de laisser planter tout le run.

    Réutilisée à la fois par le CLI de ce fichier (prétraitement en masse pour
    l'entraînement, en local) et par main.py (prétraitement à la volée du test set,
    dans le container de soumission — où logger quoi que ce soit sur des examens de
    test individuels est interdit par le règlement).

    `crop_size`/`target_spacing` DOIVENT correspondre à ce qui a été utilisé pour
    produire le cache d'entraînement (cf. train.py, qui les sauvegarde désormais
    dans results/fold_*/config.json — c'est de là que main.py les relit à
    l'inférence plutôt que de dupliquer les valeurs par défaut de ce module).

    `show_progress=False` désactive la barre tqdm : utile en particulier dans le
    container de soumission, où la sortie est redirigée vers un fichier de log non
    interactif (tqdm y imprime alors une ligne par mise à jour au lieu de rafraîchir
    une seule ligne, ce qui peut dépasser la limite de 300 lignes de log sur un test
    set de taille réelle).
    `quiet=True` désactive les logs d'erreur par fichier (voir process_single_file).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    if not files:
        return results

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(process_single_file, f, out_dir, crop_size, quiet, target_spacing): _uid_from_path(f)
            for f in files
        }
        iterator = tqdm(futures, desc="Prétraitement des DaTscans") if show_progress else futures
        for future in iterator:
            uid = futures[future]
            try:
                results[uid] = bool(future.result())
            except Exception as e:
                # Filet de sécurité supplémentaire : même une exception qui remonterait
                # jusqu'ici (crash du worker, etc.) est convertie en échec plutôt que
                # de faire planter tout le run.
                if not quiet:
                    logger.error(f"Exception non interceptée pour {uid} : {e}")
                results[uid] = False

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline CLI de traitement DaTscan robuste avec atténuation gaussienne physique."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Dossier contenant les NIfTI.")
    parser.add_argument("--cache_dir", type=str, required=True, help="Dossier pour les fichiers de sortie .npy.")
    parser.add_argument("--crop-size", type=int, nargs=3, default=list(DEFAULT_CROP_SIZE), help="Taille du cube (X Y Z).")
    parser.add_argument("--target-spacing", type=float, nargs=3, default=list(DEFAULT_TARGET_SPACING), help="Spacing isotrope cible en mm (X Y Z).")
    parser.add_argument("--jobs", type=int, default=4, help="Nombre de processus parallélisés.")

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.cache_dir)
    crop_size = tuple(args.crop_size)
    target_spacing = tuple(args.target_spacing)

    if not input_path.is_dir():
        logger.critical(f"Dossier d'entrée introuvable : {input_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(
        [p for p in input_path.glob("**/*") if p.is_file() and p.name.endswith((".nii", ".nii.gz"))]
    )

    if not nifti_files:
        logger.warning("Aucun fichier NIfTI (.nii ou .nii.gz) trouvé.")
        return

    logger.info(f"Début du traitement. {len(nifti_files)} fichiers détectés. Sortie : {output_path}")

    results = preprocess_files(nifti_files, output_path, crop_size=crop_size, target_spacing=target_spacing, jobs=args.jobs)
    success_count = sum(1 for ok in results.values() if ok)

    logger.success(f"Terminé ! {success_count}/{len(nifti_files)} fichiers convertis en .npy avec succès.")


if __name__ == "__main__":
    main()
