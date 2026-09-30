python3 resample_low_res_sites.py \
    --input-dir /neurospin/tmp/ad279118/data_ad279118_pd/train_set/niftis \
    --site-assignments /neurospin/tmp/ad279118/data_ad279118_pd/train_set/site_assignments.csv \
    --output-dir /neurospin/tmp/ad279118/data_ad279118_pd/resample_train_set \
    --better-than 2.5 2.5 2.5 \
    --jobs 8 

Lecture des résolutions natives: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1365/1365 [00:01<00:00, 809.39it/s]
2026-09-06 10:37:12.829 | WARNING  | __main__:main:470 - 3 NIfTI de /neurospin/tmp/ad279118/data_ad279118_pd/train_set/niftis n'ont pas de correspondance dans /neurospin/tmp/ad279118/data_ad279118_pd/train_set/site_assignments.csv (uid absent) -- ignorés.
2026-09-06 10:37:12.834 | INFO     | __main__:determine_small_site_targets:186 - Effectifs par site :
site
1     32
2     52
3     10
4    207
5    679
6     39
7      9
8    325
9      9
Name: count, dtype: int64
2026-09-06 10:37:12.834 | INFO     | __main__:determine_small_site_targets:187 - Seuil 'site peu représenté' (nb sujets) : < 100
2026-09-06 10:37:12.835 | INFO     | __main__:determine_small_site_targets:213 - Cible retenue -> site 2 : spacing médian (1.63, 1.63, 3.0) (52 sujets natifs)
2026-09-06 10:37:12.835 | INFO     | __main__:determine_small_site_targets:213 - Cible retenue -> site 6 : spacing médian (3.3, 3.3, 3.3) (39 sujets natifs)
2026-09-06 10:37:12.836 | INFO     | __main__:determine_small_site_targets:213 - Cible retenue -> site 1 : spacing médian (1.47, 1.47, 1.5) (32 sujets natifs)
2026-09-06 10:37:12.836 | INFO     | __main__:determine_small_site_targets:213 - Cible retenue -> site 3 : spacing médian (2.0, 2.0, 4.0) (10 sujets natifs)
2026-09-06 10:37:12.837 | INFO     | __main__:determine_small_site_targets:213 - Cible retenue -> site 7 : spacing médian (3.59, 3.59, 3.59) (9 sujets natifs)
2026-09-06 10:37:12.838 | INFO     | __main__:determine_small_site_targets:213 - Cible retenue -> site 9 : spacing médian (4.42, 4.42, 4.42) (9 sujets natifs)
2026-09-06 10:37:12.838 | INFO     | __main__:build_jobs:291 - 916/1362 sujets ont une résolution native meilleure que (2.5, 2.5, 2.5) sur les 3 axes -> éligibles à la dégradation synthétique.
2026-09-06 10:37:12.917 | INFO     | __main__:build_jobs:318 - 2812 rééchantillonnages synthétiques à générer.
Rééchantillonnage synthétique: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2812/2812 [03:46<00:00, 12.41it/s]
2026-09-06 10:40:59.564 | SUCCESS  | __main__:main:532 - Terminé. 2812/2812 NIfTI synthétiques écrits dans /neurospin/tmp/ad279118/data_ad279118_pd/resample_train_set/niftis. CSV combiné (réel + synthétique) : /neurospin/tmp/ad279118/data_ad279118_pd/resample_train_set/site_assignments.csv (4174 lignes).
2026-09-06 10:40:59.564 | INFO     | __main__:main:536 - Pour l'entraînement : concaténer les dossiers de NIfTI natifs et /neurospin/tmp/ad279118/data_ad279118_pd/resample_train_set/niftis avant de lancer preprocess_dataset.py, et utiliser get_stratified_group_splits (group_col='source_uid') plutôt que get_stratified_splits pour construire les folds, afin qu'un même sujet (natif + variantes synthétiques) reste toujours dans un seul et même fold.


python3 my_module/preprocess_dataset.py \
    --input_dir /neurospin/tmp/ad279118/data_ad279118_pd/resample_train_set/niftis \
    --cache_dir /neurospin/tmp/ad279118/data_ad279118_pd/resample_train_set/80iso_r_cache_dir \
    --jobs 24

2026-09-06 10:46:38.585 | INFO     | __main__:main:414 - Début du traitement. 2812 fichiers détectés. Sortie : /neurospin/tmp/ad279118/data_ad279118_pd/resample_train_set/80iso_r_cache_dir
Prétraitement des DaTscans: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2812/2812 [06:35<00:00,  7.10it/s]
2026-09-06 10:53:14.711 | SUCCESS  | __main__:main:419 - Terminé ! 2812/2812 fichiers convertis en .npy avec succès.



python3 my_module/preprocess_dataset.py \
    --input_dir /neurospin/tmp/ad279118/data_ad279118_pd/train_set/niftis \
    --cache_dir /neurospin/tmp/ad279118/data_ad279118_pd/resample_train_set/80iso_r_cache_dir \
    --jobs 24

2026-09-06 10:53:37.912 | INFO     | __main__:main:414 - Début du traitement. 1365 fichiers détectés. Sortie : /neurospin/tmp/ad279118/data_ad279118_pd/resample_train_set/80iso_r_cache_dir
Prétraitement des DaTscans: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1365/1365 [06:51<00:00,  3.32it/s]
2026-09-06 11:00:29.131 | SUCCESS  | __main__:main:419 - Terminé ! 1365/1365 fichiers convertis en .npy avec succès.


ls /neurospin/tmp/ad279118/data_ad279118_pd/resample_train_set/80iso_r_cache_dir/* | wc -l
4177


python3 /neurospin/tmp/ad279118/data_ad279118_pd/scripts/resnet_resample/train.py \
--train_labels /neurospin/tmp/ad279118/data_ad279118_pd/resample_train_set/site_assignments.csv \
--cache_dir /neurospin/tmp/ad279118/data_ad279118_pd/resample_train_set/80iso_r_cache_dir   \
--model ResNet \
--depth resnet18 \
--base_channels 16  \
--attention cbam \
--mlp_layers 1 \
--pool_mode max  \
--batch_size 32 \
--epochs 300 \
--patience 30 \
--lr 0.001  \
--max_rotate_deg 40 --max_translate_vox 20 \
--num_workers 19 \
--seed 3 --loss bce --dropout 0.25 \
--results_dir /neurospin/tmp/ad279118/data_ad279118_pd/ResNet18_res22 \
--lr_patience 15 \
--lr_factor 0.3 \
--p_noise 0.95 --noise_std 0.12 \
--max_scale_delta 0.15 \
--p_elastic 0.5 --elastic_alpha 5  \
--p_shift_intensity 0.2  --intensity_shift_delta 0.2 \
--p_blur 0.55 --dropout_holes 5 \
--dropout_frac 0.09 --blur_sigma_max 0.8
