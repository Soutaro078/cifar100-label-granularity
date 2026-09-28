import torchvision

from utils.cifar import load_cifar100_meta, load_split_dict
from utils.paths import DATA_ROOT
from utils.transforms import cifar100_eval_transform

transform = cifar100_eval_transform()
trainset = torchvision.datasets.CIFAR100(
    root=str(DATA_ROOT),
    train=True,
    download=True,
    transform=transform,
)

print(f"データの総数: {len(trainset)}枚\n")

meta = load_cifar100_meta()
fine_classes = trainset.classes
coarse_classes = meta["coarse_label_names"]
coarse_labels = load_split_dict("train")["coarse_labels"]

print(f"粗いクラス（Coarse）の数: {len(coarse_classes)}個")
print(f"粗いクラスの一例: {coarse_classes[:5]}\n")

image, fine_label_idx = trainset[0]
coarse_label_idx = coarse_labels[0]

print("--- データの1例 ---")
print(f"画像のテンソル形状: {image.shape}")
print(f"細かいラベル番号 (Fine): {fine_label_idx} -> {fine_classes[fine_label_idx]}")
print(f"粗いラベル番号 (Coarse): {coarse_label_idx} -> {coarse_classes[coarse_label_idx]}")

target_coarse_name = "trees"
target_coarse_idx = coarse_classes.index(target_coarse_name)
tree_indices = [i for i, label in enumerate(coarse_labels) if label == target_coarse_idx]

print(f"\n[確認] 粗いクラス '{target_coarse_name}' に属する画像の枚数: {len(tree_indices)}枚")
