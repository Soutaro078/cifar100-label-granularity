import torch

from models.resnet import assert_cifar32_stem, build_cifar_resnet18

print("--- 1. ResNet-18の読み込みと確認 ---")

num_classes = 5
resnet18 = build_cifar_resnet18(num_classes)
assert_cifar32_stem(resnet18)

conv = resnet18.conv1
print("ResNet-18（CIFAR 32×32 用）の構築が完了しました。")
print(
    f"conv1: kernel={conv.kernel_size}, stride={conv.stride}, "
    f"padding={conv.padding}, bias={conv.bias is not None}"
)
print(f"maxpool: {type(resnet18.maxpool).__name__}\n")

print("--- 2. 入力と出力の形状（シェイプ）のテスト ---")
dummy_input = torch.randn(4, 3, 32, 32)
print(f"入力データの形状 (Batch, Channel, Height, Width): {dummy_input.shape}")

output = resnet18(dummy_input)
print(f"出力データの形状 (Batch, Num_Classes): {output.shape}")
print("-> 期待値: torch.Size([4, 5]) （バッチサイズ4、5クラス分のスコア）")

if output.shape == torch.Size([4, 5]):
    print("\n[判定] 成功！ResNet-18のインポートと形状確認はバッチリです。")
else:
    print("\n[判定] 形状が一致しません。確認が必要です。")
