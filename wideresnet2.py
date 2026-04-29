"""
Wide Residual Networks (WRN) — PyTorch
=======================================
Implémentation fidèle au papier original :
  "Wide Residual Networks" — Zagoruyko & Komodakis, BMVC 2016
  https://arxiv.org/abs/1605.07146
 
Architecture (pré-activation, BN → ReLU → Conv) :
  - Bloc de base  : BN → ReLU → Conv(3×3) → BN → ReLU → [Dropout] → Conv(3×3)
  - Skip connection avec projection 1×1 si les dimensions changent
  - 3 groupes de blocs larges (largeur × k)
  - Global average pooling → FC
 
Notation du papier : WRN-{depth}-{widen_factor}
  Exemples courants : WRN-28-10, WRN-40-4, WRN-16-8
"""
 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
# ---------------------------------------------------------------------------
# Bloc résiduel de base (pré-activation)
# ---------------------------------------------------------------------------
 
class BasicBlock(nn.Module):
    """
    Bloc B(3,3) du papier : deux convolutions 3×3 avec pré-activation.
    Le dropout (optionnel) est placé entre les deux convolutions,
    exactement comme dans le code original de Zagoruyko.
    """
 
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, dropout_rate: float = 0.0):
        super().__init__()
 
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
 
        self.bn2      = nn.BatchNorm2d(out_channels)
        self.relu2    = nn.ReLU(inplace=True)
        self.dropout  = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None
        self.conv2    = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                  stride=1, padding=1, bias=False)
 
        # Projection 1×1 si la dimension change (stride ou nombre de canaux)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                      stride=stride, bias=False)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pré-activation
        out = self.relu1(self.bn1(x))
 
        # Skip connection calculée depuis la sortie pré-activation
        shortcut = self.shortcut(out) if self.shortcut is not None else x
 
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv2(out)
 
        return out + shortcut
 
 
# ---------------------------------------------------------------------------
# Groupe de blocs (NetworkBlock dans le code original)
# ---------------------------------------------------------------------------
 
class NetworkBlock(nn.Module):
    """
    Empilement de `num_blocks` BasicBlock avec le même nombre de canaux.
    Seul le premier bloc peut avoir stride > 1 (sous-échantillonnage).
    """
 
    def __init__(self, num_blocks: int, in_channels: int, out_channels: int,
                 stride: int, dropout_rate: float = 0.0):
        super().__init__()
        layers = []
        for i in range(num_blocks):
            layers.append(BasicBlock(
                in_channels  = in_channels  if i == 0 else out_channels,
                out_channels = out_channels,
                stride       = stride       if i == 0 else 1,
                dropout_rate = dropout_rate,
            ))
        self.layer = nn.Sequential(*layers)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)
 
 
# ---------------------------------------------------------------------------
# WideResNet principal
# ---------------------------------------------------------------------------
 
class WideResNet(nn.Module):
    """
    WRN-{depth}-{widen_factor} pour CIFAR-10 / CIFAR-100.
 
    Paramètres
    ----------
    depth         : profondeur totale (doit vérifier (depth - 4) % 6 == 0)
    widen_factor  : facteur d'élargissement k
    num_classes   : nombre de classes de sortie (10 ou 100 pour CIFAR)
    dropout_rate  : taux de dropout dans chaque bloc (0 = désactivé)
    """
 
    def __init__(self, depth: int = 28, widen_factor: int = 10,
                 num_classes: int = 10, dropout_rate: float = 0.0):
        super().__init__()
 
        assert (depth - 4) % 6 == 0, \
            f"La profondeur doit vérifier (depth - 4) % 6 == 0, reçu {depth}."
 
        n = (depth - 4) // 6          # nombre de blocs par groupe
        k = widen_factor
 
        # Largeurs des 3 groupes : [16, 16·k, 32·k, 64·k]
        nChannels = [16, 16 * k, 32 * k, 64 * k]
 
        # Première convolution 3×3 (pas de BN avant, contrairement aux blocs)
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3,
                               stride=1, padding=1, bias=False)
 
        # Trois groupes de blocs larges
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1],
                                   stride=1, dropout_rate=dropout_rate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2],
                                   stride=2, dropout_rate=dropout_rate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3],
                                   stride=2, dropout_rate=dropout_rate)
 
        # BN + ReLU finale (pré-activation : les blocs ne terminent pas par BN)
        self.bn_final   = nn.BatchNorm2d(nChannels[3])
        self.relu_final = nn.ReLU(inplace=True)
 
        # Classificateur
        self.fc = nn.Linear(nChannels[3], num_classes)
 
        self.nChannels = nChannels
 
        # Initialisation des poids (He / Kaiming)
        self._initialize_weights()
 
    # ------------------------------------------------------------------
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
 
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu_final(self.bn_final(out))
        # Global average pooling
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)
 
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Retourne le nombre de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)