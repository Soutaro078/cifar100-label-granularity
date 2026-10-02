"""CIFAR 向け ResNet-18。

32×32 入力では ImageNet 用の stem（7×7 conv, stride 2 + maxpool）だと
特徴マップが小さくなりすぎる。ここでの仕様は次のとおり。

- conv1: 3×3, stride 1, padding 1, bias なし
- maxpool: Identity（プーリングしない）
"""

import torch.nn as nn
import torchvision


def build_cifar_resnet18(num_classes: int) -> nn.Module:
    """32×32 用に stem を差し替えた ResNet-18 を返す。"""
    if num_classes < 1:
        raise ValueError(f"num_classes は 1 以上である必要があります: {num_classes}")

    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # torchvision の ReLU は inplace=True。学習中に出力を集計すると
    # CUDA illegal memory access になることがある。
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
    assert_cifar32_stem(model)
    return model


def assert_cifar32_stem(model: nn.Module) -> None:
    """第1畳み込みと maxpool が 32×32 用のままか確認する。"""
    conv = model.conv1
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f"conv1 が Conv2d ではありません: {type(conv).__name__}")
    if conv.kernel_size != (3, 3) or conv.stride != (1, 1) or conv.padding != (1, 1):
        raise ValueError(
            "CIFAR 32×32 用の conv1 ではありません: "
            f"kernel={conv.kernel_size}, stride={conv.stride}, padding={conv.padding}"
        )
    if conv.in_channels != 3 or conv.out_channels != 64 or conv.bias is not None:
        raise ValueError(
            "conv1 は in=3, out=64, bias=False である必要があります: "
            f"in={conv.in_channels}, out={conv.out_channels}, bias={conv.bias is not None}"
        )
    if not isinstance(model.maxpool, nn.Identity):
        raise TypeError(
            "CIFAR 32×32 用には maxpool を Identity にする必要があります: "
            f"{type(model.maxpool).__name__}"
        )
