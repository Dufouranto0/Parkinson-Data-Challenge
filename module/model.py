"""
3D-ResNet léger (ResNet-10 / ResNet-18 adapté en 3D) avec blocs résiduels et
modules d'attention (Squeeze-and-Excitation ou CBAM) optionnels.

Fichier volontairement séparé de model.py : les utilitaires génériques (FocalLoss,
calibration par température, collect_logits, evaluate_metrics...) restent dans
model.py et s'appliquent indifféremment à cette architecture ou à Conv3DNet -- ce
fichier ne contient QUE la définition du réseau.

Contrairement à un ResNet "image naturelle" (stem agressif : conv 7x7 stride 2 +
maxpool), le stem ici ne downsample pas : le volume d'entrée est déjà petit
(crop_size typiquement 80^3 voxels), on ne veut pas jeter de détail spatial avant
même d'avoir commencé. Le downsampling (x2 à chaque fois) est porté par les 3
dernières des 4 stages, comme dans un ResNet classique.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score

class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation : attention canal uniquement (pas de localisation
    spatiale). Un pooling global par canal, un petit MLP goulot, un gate sigmoïde
    qui rescale chaque canal."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class ChannelAttention3D(nn.Module):
    """Composante canal de CBAM : comme SE, mais combine descripteurs avg ET max
    poolés (à travers un MLP à poids partagés) plutôt que la seule moyenne."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        gate = self.sigmoid(avg_out + max_out).view(b, c, 1, 1, 1)
        return x * gate


class SpatialAttention3D(nn.Module):
    """Composante spatiale de CBAM : moyenne/max à travers les canaux -> carte 2D
    (ici 3D) -> conv -> gate sigmoïde appliqué à chaque voxel. C'est cette
    composante qui manque à SE et qui permet d'apprendre "quelle région du volume
    regarder" (ex: striatum gauche vs droit), pas seulement "quels canaux"."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        gate = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * gate


class CBAM3D(nn.Module):
    """Convolutional Block Attention Module : attention canal puis spatiale,
    appliquées séquentiellement (ordre standard de l'article de Woo et al.)."""
    def __init__(self, channels: int, reduction: int = 8, spatial_kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention3D(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention3D(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


def _build_attention(channels: int, attention: str, se_reduction: int, cbam_reduction: int, cbam_spatial_kernel: int) -> nn.Module:
    if attention == "none":
        return nn.Identity()
    if attention == "se":
        return SEBlock3D(channels, reduction=se_reduction)
    if attention == "cbam":
        return CBAM3D(channels, reduction=cbam_reduction, spatial_kernel_size=cbam_spatial_kernel)
    raise ValueError(f"attention inconnu: {attention!r} (attendu: 'none', 'se', 'cbam')")


class BasicBlock3D(nn.Module):
    """Bloc résiduel standard (2x conv3x3x3 + BN + activation), avec module
    d'attention optionnel appliqué sur la branche résiduelle juste avant l'addition
    avec le raccourci -- placement standard SENet/CBAM. Le raccourci passe par une
    projection 1x1x1 + BN dès que le stride ou le nombre de canaux changent."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        attention: str = "none",
        se_reduction: int = 8,
        cbam_reduction: int = 8,
        cbam_spatial_kernel: int = 7,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

        self.attention = _build_attention(out_channels, attention, se_reduction, cbam_reduction, cbam_spatial_kernel)

        self.downsample: Optional[nn.Module] = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.activation(out)
        return out


class ResNet3D(nn.Module):
    """3D-ResNet léger à 4 stages. `layers=(1,1,1,1)` -> ResNet-10 (1 bloc/stage),
    `layers=(2,2,2,2)` -> ResNet-18 (2 blocs/stage) -- construire via build_model
    plutôt que d'appeler cette classe directement."""
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        layers: tuple = (1, 1, 1, 1),
        attention: str = "cbam",
        se_reduction: int = 8,
        cbam_reduction: int = 8,
        cbam_spatial_kernel: int = 7,
        mlp_layers: int = 2,
        mlp_hidden: int = 64,
        dropout: float = 0.3,
        pool_mode: str = "avg_max",  # "avg" | "max" | "avg_max" -- même convention que Conv3DNet
    ):
        super().__init__()
        self.pool_mode = pool_mode

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )

        stage_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        stage_strides = [1, 2, 2, 2]  # la 1ere stage ne downsample pas, les 3 suivantes oui

        stages = []
        in_ch = base_channels
        for out_ch, stride, n_blocks in zip(stage_channels, stage_strides, layers):
            stage_blocks = [
                BasicBlock3D(
                    in_ch, out_ch, stride=stride, attention=attention,
                    se_reduction=se_reduction, cbam_reduction=cbam_reduction, cbam_spatial_kernel=cbam_spatial_kernel,
                )
            ]
            for _ in range(1, n_blocks):
                stage_blocks.append(
                    BasicBlock3D(
                        out_ch, out_ch, stride=1, attention=attention,
                        se_reduction=se_reduction, cbam_reduction=cbam_reduction, cbam_spatial_kernel=cbam_spatial_kernel,
                    )
                )
            stages.append(nn.Sequential(*stage_blocks))
            in_ch = out_ch

        self.stages = nn.ModuleList(stages)
        self.out_channels = in_ch

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        pooled_channels = self.out_channels * 2 if pool_mode == "avg_max" else self.out_channels

        if mlp_layers == 1:
            self.mlp = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(pooled_channels, 1),
            )
        elif mlp_layers == 2:
            self.mlp = nn.Sequential(
                nn.Linear(pooled_channels, mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, 1),
            )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool_mode == "avg":
            return self.global_avg_pool(x).flatten(1)
        if self.pool_mode == "max":
            return self.global_max_pool(x).flatten(1)
        return torch.cat([self.global_avg_pool(x).flatten(1), self.global_max_pool(x).flatten(1)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.mlp(self._pool(x)).squeeze(1)


_LAYERS_BY_DEPTH = {
    "resnet10": (1, 1, 1, 1),
    "resnet18": (2, 2, 2, 2),
}


def build_model(
    base_channels: int = 32,
    depth: str = "resnet10",
    attention: str = "cbam",
    se_reduction: int = 8,
    cbam_reduction: int = 8,
    cbam_spatial_kernel: int = 7,
    mlp_layers: int = 2,
    mlp_hidden: int = 64,
    dropout: float = 0.3,
    pool_mode: str = "avg_max",
) -> ResNet3D:
    if depth not in _LAYERS_BY_DEPTH:
        raise ValueError(f"depth inconnu: {depth!r} (attendu: {list(_LAYERS_BY_DEPTH)})")

    return ResNet3D(
        in_channels=1,
        base_channels=base_channels,
        layers=_LAYERS_BY_DEPTH[depth],
        attention=attention,
        se_reduction=se_reduction,
        cbam_reduction=cbam_reduction,
        cbam_spatial_kernel=cbam_spatial_kernel,
        mlp_layers=mlp_layers,
        mlp_hidden=mlp_hidden,
        dropout=dropout,
        pool_mode=pool_mode,
    )
    
    

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) pour classification binaire à partir de logits.

    Sert à limiter le poids des exemples faciles/bien classés (majoritaires dans un
    dataset de DaTscans) dans le gradient, au profit des cas difficiles ou ambigus —
    typiquement les examens pathologiques sans anomalie visible à l'œil du
    clinicien, où le modèle doit apprendre à repérer des indices plus subtils que
    dans la majorité des cas nets.

    Deux paramètres à rôles distincts :
    - `gamma` : plus il est grand, plus les exemples déjà bien classés (p_t proche
      de 1) sont atténués dans la perte. 2.0 est la valeur standard de l'article
      original ; c'est le paramètre qui répond directement au problème "beaucoup de
      cas faciles, une poignée de cas difficiles".
    - `alpha` : rééquilibrage de PRÉVALENCE entre les deux classes (poids relatif de
      la classe positive), un rôle différent de gamma. Laisser à 0.5 si les classes
      sont déjà globalement équilibrées, ou l'ajuster (ex: 0.25) sinon.

    Ne remplace la BCEWithLogitsLoss QUE comme objectif d'optimisation : la
    sélection du meilleur checkpoint (et le scheduler LR) dans train.py restent
    basés sur la val log loss (sklearn.metrics.log_loss), indépendamment de la loss
    utilisée ici pour la descente de gradient.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        # p_t : probabilité assignée par le modèle à la classe réellement observée.
        # Fonctionne aussi avec des targets adoucies par label_smoothing (valeurs
        # dans ]0,1[ plutôt que strictement 0/1), comme le reste du pipeline.
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def fit_temperature(logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, max_iter: int = 200) -> TemperatureScaler:
    scaler = TemperatureScaler()
    logits_t = torch.from_numpy(logits.astype(np.float32))
    labels_t = torch.from_numpy(labels.astype(np.float32))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits_t), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler

def fit_temperature_robust(logits: np.ndarray, labels: np.ndarray, n_bootstrap: int = 50, seed: int = 42) -> float:
    if n_bootstrap <= 1:
        return float(fit_temperature(logits, labels).temperature.item())
    rng = np.random.default_rng(seed)
    n = len(logits)
    log_temps = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scaler = fit_temperature(logits[idx], labels[idx])
        log_temps.append(np.log(scaler.temperature.item()))
    return float(np.exp(np.mean(log_temps)))

@torch.no_grad()
def collect_logits(model: nn.Module, loader, device: str) -> tuple[np.ndarray, np.ndarray, list]:
    """Récupère les logits et étiquettes depuis un DataLoader émettant des triplets."""
    model.eval()
    all_logits, all_labels, all_uids = [], [], []
    # AJUSTEMENT : Prise en compte du triplet (x, y, uid) retourné par le dataset
    for x, y, uid in loader:
        logits = model(x.to(device)).cpu().numpy()
        all_logits.append(logits)
        all_labels.append(y.numpy())
        all_uids.extend(uid)
    return np.concatenate(all_logits), np.concatenate(all_labels), all_uids

@torch.no_grad()
def collect_logits_unlabeled(model: nn.Module, loader, device: str) -> tuple[np.ndarray, list]:
    model.eval()
    all_logits, all_uids = [], []
    for x, uid in loader:
        logits = model(x.to(device)).cpu().numpy()
        all_logits.append(logits)
        all_uids.extend(uid)
    return np.concatenate(all_logits), all_uids

def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float, rng: Optional[np.random.Generator] = None) -> tuple[torch.Tensor, torch.Tensor]:
    if alpha <= 0:
        return x, y
    lam = (rng or np.random.default_rng()).beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]

def train_one_epoch(
    model: nn.Module, 
    loader, 
    optimizer, 
    device: str, 
    criterion: Optional[nn.Module] = None,
    label_smoothing: float = 0.0, 
    mixup_alpha: float = 0.0, 
    mixup_rng: Optional[np.random.Generator] = None
) -> float:
    """Entraîne le modèle sur une époque complète en ignorant l'UID.

    `criterion` : fonction de perte utilisée pour la descente de gradient
    (BCEWithLogitsLoss par défaut si non fournie ; passer une FocalLoss ici pour
    l'utiliser à la place). Ne change rien à la façon dont le meilleur checkpoint
    est sélectionné ailleurs dans le pipeline (cf. train.py), qui reste toujours
    basée sur la val log loss.
    """
    model.train()
    criterion = criterion or nn.BCEWithLogitsLoss()
    total_loss, n = 0.0, 0
    # AJUSTEMENT : Extraction propre du triplet en ignorant explicitement l'identifiant (_)
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        x, y = mixup_batch(x, y, mixup_alpha, rng=mixup_rng)
        y_target = y * (1.0 - label_smoothing) + 0.5 * label_smoothing if label_smoothing > 0 else y
        optimizer.zero_grad()
        loss = criterion(model(x), y_target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)

def probs_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return np.clip(1.0 / (1.0 + np.exp(-(logits / temperature))), 1e-7, 1 - 1e-7)

def evaluate_metrics(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> dict:
    probs = probs_from_logits(logits, temperature=temperature)
    return {"log_loss": log_loss(labels, probs), "auroc": roc_auc_score(labels, probs)}

@dataclass
class EarlyStopping:
    patience: int = 10
    min_delta: float = 1e-4
    best_loss: float = float("inf")
    best_epoch: int = -1
    num_bad_epochs: int = 0

    def step(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            return True
        self.num_bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.num_bad_epochs >= self.patience
