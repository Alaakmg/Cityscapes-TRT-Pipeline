"""ResNet50-encoder U-Net, kept deployment-friendly on purpose:

  * plain Conv/BN/ReLU + nearest upsampling only (clean ONNX export, no custom ops)
  * F.interpolate always gets an integer scale_factor, never a dynamic size=,
    so TensorRT sees static Resize layers
  * single output head, no deep supervision

Input:  (N, 3, H, W), H and W multiples of 32.
Output: (N, num_classes, H, W) raw logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from torchvision.models import resnet50
from torchvision.models.resnet import Bottleneck


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Upsample x2 (nearest), concat skip, 2x ConvBNReLU."""

    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv1 = ConvBNReLU(cin + cskip, cout)
        self.conv2 = ConvBNReLU(cout, cout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


class ResNet50UNet(nn.Module):
    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V2" if pretrained else None
        r = resnet50(weights=weights)

        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)  # /2,   64ch
        self.pool = r.maxpool                              # /4
        self.layer1 = r.layer1                             # /4,  256ch
        self.layer2 = r.layer2                             # /8,  512ch
        self.layer3 = r.layer3                             # /16, 1024ch
        self.layer4 = r.layer4                             # /32, 2048ch

        self.dec4 = DecoderBlock(2048, 1024, 512)          # /16
        self.dec3 = DecoderBlock(512, 512, 256)            # /8
        self.dec2 = DecoderBlock(256, 256, 128)            # /4
        self.dec1 = DecoderBlock(128, 64, 64)              # /2
        self.dec0 = DecoderBlock(64, 0, 32)                # /1
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.stem(x)              # /2
        s2 = self.layer1(self.pool(s1))  # /4
        s3 = self.layer2(s2)           # /8
        s4 = self.layer3(s3)           # /16
        bottom = self.layer4(s4)       # /32

        d = self.dec4(bottom, s4)
        d = self.dec3(d, s3)
        d = self.dec2(d, s2)
        d = self.dec1(d, s1)
        d = self.dec0(d, None)
        return self.head(d)


def build_model(num_classes: int = 8, pretrained: bool = True) -> ResNet50UNet:
    return ResNet50UNet(num_classes=num_classes, pretrained=pretrained)


def fold_batchnorm(model: nn.Module) -> int:
    """Fold every BatchNorm2d into the conv that precedes it (exact in eval mode).

    Must run on the plain model *before* mtq.quantize. ModelOpt's QuantConv2d
    is not a plain conv, so torch's ONNX exporter can't fold BN into it; the
    BN then sits between conv3 and the residual add in the exported graph and
    breaks TensorRT's conv+add+ReLU INT8 fusion (measured on the Jetson: the
    QAT v2 engine kept 129 layers vs PTQ's 78). Folding first gives TensorRT
    the conv -> add -> ReLU pattern it expects.
    """
    model.eval()
    n = 0
    for m in model.modules():
        if isinstance(m, Bottleneck):  # torchvision ResNet block
            for c, b in (("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3")):
                setattr(m, c, fuse_conv_bn_eval(getattr(m, c), getattr(m, b)))
                setattr(m, b, nn.Identity())
                n += 1
        if isinstance(m, nn.Sequential):  # stem, downsample, decoder ConvBNReLU
            names = list(m._modules)
            for i in range(len(names) - 1):
                a, b = m._modules[names[i]], m._modules[names[i + 1]]
                if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                    m._modules[names[i]] = fuse_conv_bn_eval(a, b)
                    m._modules[names[i + 1]] = nn.Identity()
                    n += 1
    return n
