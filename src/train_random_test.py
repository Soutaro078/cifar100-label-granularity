import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset

from models.resnet import build_cifar_resnet18
from utils.cifar import load_cifar100_meta, load_split_dict
from utils.paths import DATA_ROOT
from utils.transforms import cifar100_eval_transform

print("--- 1. データの準備（特定の粗いクラス 'trees' の5クラスを抽出） ---")

transform = cifar100_eval_transform()
trainset = torchvision.datasets.CIFAR100(
    root=str(DATA_ROOT), train=True, download=True, transform=transform
)

coarse_classes = load_cifar100_meta()["coarse_label_names"]
coarse_labels = load_split_dict("train")["coarse_labels"]

target_coarse_name = "trees"
target_coarse_idx = coarse_classes.index(target_coarse_name)
tree_indices = [i for i, label in enumerate(coarse_labels) if label == target_coarse_idx]

tree_subset = Subset(trainset, tree_indices)
train_loader = DataLoader(tree_subset, batch_size=32, shuffle=True)

print(f"'{target_coarse_name}' のデータ数: {len(tree_subset)}枚のサブセットを作成しました。")


print("\n--- 2. モデルの準備（ランダム初期化のResNet-18・5クラス分類用） ---")
resnet18 = build_cifar_resnet18(num_classes=5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18 = resnet18.to(device)
print(f"使用デバイス: {device}")


print("\n--- 3. 動作確認用の学習ループテスト（数バッチだけ回す） ---")
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(), lr=0.01, momentum=0.9)

resnet18.train()

for batch_idx, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    outputs = resnet18(images)

    dummy_labels = torch.randint(0, 5, (images.size(0),), device=device)
    loss = criterion(outputs, dummy_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx == 4:
        print(f"バッチ {batch_idx + 1} まで正常に学習ループが回りました！")
        print(f"現在の損失 (Loss): {loss.item():.4f}")
        break

print("\n[判定] 成功！ランダム初期化モデルでの学習ループ（順伝播・逆伝播）が正常に動作しました。")
