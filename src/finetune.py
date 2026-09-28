import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset

from models.resnet import build_cifar_resnet18
from utils.cifar import load_cifar100_meta, load_split_dict
from utils.config import apply_overrides, load_yaml
from utils.paths import CONFIGS_DIR, DATA_ROOT, pretrain_checkpoint_path
from utils.tracking import finish_wandb, init_wandb, log_metrics, wandb_settings
from utils.transforms import cifar100_eval_transform


class RemappedSubsetDataset(Dataset):
    """ラベルを 0 から始まる局所 ID に付け替える。"""

    def __init__(self, dataset, indices, fine_to_local_map):
        self.dataset = dataset
        self.indices = indices
        self.fine_to_local_map = fine_to_local_map

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, orig_label = self.dataset[real_idx]
        local_label = self.fine_to_local_map[orig_label]
        return img, torch.tensor(local_label, dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-100 Fine-tuning & Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIGS_DIR / "finetune.yaml"),
        help="ハイパーパラメータの YAML",
    )
    parser.add_argument(
        "--init_mode",
        type=str,
        default=None,
        choices=["fine", "coarse", "random"],
        help="初期化モデルの種類: 'fine', 'coarse', または 'random'",
    )
    parser.add_argument(
        "--target_group",
        type=str,
        default=None,
        help="検証する粗いクラスの名前（例: 'trees', 'vehicles_1'）",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=None,
        help="1クラスあたりの学習データ数",
    )
    parser.add_argument("--epochs", type=int, default=None, help="ファインチューニングのエポック数")
    parser.add_argument("--batch_size", type=int, default=None, help="バッチサイズ")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B のプロジェクト名")
    parser.add_argument("--no_wandb", action="store_true", help="W&B への記録を止める")
    return parser.parse_args()


def build_config(args) -> dict:
    return apply_overrides(
        load_yaml(args.config),
        {
            "init_mode": args.init_mode,
            "target_group": args.target_group,
            "samples_per_class": args.samples_per_class,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
    )


def load_pretrained_backbone(model, init_mode: str, device: torch.device) -> None:
    if init_mode == "random":
        print("ランダム初期化モデルを使用します（事前学習なし）。")
        return

    weights_path = pretrain_checkpoint_path(init_mode)
    label = "Fine事前学習モデル" if init_mode == "fine" else "Coarse事前学習モデル"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"{label}が見つかりません: {weights_path}\n"
            f"先に python src/pretrain.py --mode {init_mode} を実行してください。"
        )
    print(f"{label}をロードします: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict.pop("fc.weight", None)
    state_dict.pop("fc.bias", None)
    model.load_state_dict(state_dict, strict=False)


def main():
    args = parse_args()
    cfg = build_config(args)
    init_mode = cfg["init_mode"]
    target_group = cfg["target_group"]
    samples_per_class = int(cfg["samples_per_class"])
    epochs = int(cfg["epochs"])
    batch_size = int(cfg["batch_size"])

    print("=== ファインチューニング実験 ===")
    print(
        f"対象グループ: {target_group} | 初期化: {init_mode.upper()} | "
        f"サンプル数: {samples_per_class}枚/クラス\n"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # サンプル効率の比較では、学習時も評価時も拡張なしの正規化だけを使う。
    transform = cifar100_eval_transform()

    trainset = torchvision.datasets.CIFAR100(
        root=str(DATA_ROOT), train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR100(
        root=str(DATA_ROOT), train=False, download=True, transform=transform
    )

    meta = load_cifar100_meta()
    fine_classes = meta["fine_label_names"]
    coarse_classes = meta["coarse_label_names"]
    train_dict = load_split_dict("train")

    if target_group not in coarse_classes:
        raise ValueError(
            f"指定されたグループ '{target_group}' は CIFAR-100 の Coarse クラスに存在しません。\n"
            f"利用可能リスト: {coarse_classes}"
        )

    target_coarse_idx = coarse_classes.index(target_group)
    target_fine_classes = set()
    for img_idx, coarse_idx in enumerate(train_dict["coarse_labels"]):
        if coarse_idx == target_coarse_idx:
            target_fine_classes.add(trainset.targets[img_idx])
    target_fine_classes = sorted(target_fine_classes)

    print(f"対象グループ '{target_group}' に属する細かいクラス（計 {len(target_fine_classes)}個）:")
    for local_id, fine_idx in enumerate(target_fine_classes):
        print(f"  Local ID {local_id}: {fine_classes[fine_idx]}")

    fine_to_local = {fine_idx: local_id for local_id, fine_idx in enumerate(target_fine_classes)}
    num_classes = len(target_fine_classes)

    random.seed(int(cfg["seed"]))
    class_to_indices = {fine_idx: [] for fine_idx in target_fine_classes}
    for idx, target in enumerate(trainset.targets):
        if target in class_to_indices:
            class_to_indices[target].append(idx)

    selected_train_indices = []
    for fine_idx in target_fine_classes:
        indices = class_to_indices[fine_idx]
        sampled = random.sample(indices, min(len(indices), samples_per_class))
        selected_train_indices.extend(sampled)

    train_subset = RemappedSubsetDataset(trainset, selected_train_indices, fine_to_local)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    selected_test_indices = [
        idx for idx, target in enumerate(testset.targets) if target in fine_to_local
    ]
    test_subset = RemappedSubsetDataset(testset, selected_test_indices, fine_to_local)
    test_loader = DataLoader(
        test_subset, batch_size=int(cfg["test_batch_size"]), shuffle=False
    )

    print(f"\n訓練データ数（少数絞り込み後）: {len(train_subset)}枚")
    print(f"テストデータ数（対象グループ）: {len(test_subset)}枚\n")

    model = build_cifar_resnet18(num_classes)
    load_pretrained_backbone(model, init_mode, device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=float(cfg["lr"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    wandb_on, wandb_project = wandb_settings(cfg, args.no_wandb, args.wandb_project)
    run = init_wandb(
        enabled=wandb_on,
        project=wandb_project,
        name=f"finetune-{init_mode}-{target_group}-n{samples_per_class}",
        group=f"finetune-{target_group}",
        job_type="finetune",
        config={
            "stage": "finetune",
            "init_mode": init_mode,
            "target_group": target_group,
            "samples_per_class": samples_per_class,
            "epochs": epochs,
            "batch_size": batch_size,
            "test_batch_size": int(cfg["test_batch_size"]),
            "learning_rate": float(cfg["lr"]),
            "momentum": float(cfg["momentum"]),
            "weight_decay": float(cfg["weight_decay"]),
            "seed": int(cfg["seed"]),
            "architecture": "resnet18_cifar32",
            "num_classes": num_classes,
            "device": str(device),
        },
    )

    try:
        print("\n--- ファインチューニング開始 ---")
        test_loss = None
        test_acc = None
        for epoch_index in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch = epoch_index + 1
            epoch_loss = running_loss / total
            epoch_acc = 100.0 * correct / total

            model.eval()
            test_loss_sum = 0.0
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss_sum += loss.item() * labels.size(0)
                    _, predicted = outputs.max(1)
                    test_total += labels.size(0)
                    test_correct += predicted.eq(labels).sum().item()

            test_loss = test_loss_sum / test_total
            test_acc = 100.0 * test_correct / test_total
            print(
                f"Epoch [{epoch}/{epochs}] | Train Loss: {epoch_loss:.4f} | "
                f"Train Acc: {epoch_acc:.2f}% | Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc:.2f}%"
            )
            log_metrics(
                run,
                {
                    "epoch": epoch,
                    "finetune/train_loss": epoch_loss,
                    "finetune/train_accuracy": epoch_acc,
                    "finetune/test_loss": test_loss,
                    "finetune/test_accuracy": test_acc,
                },
            )

        print(f"\n[結果評価] テストデータ正解率 (Accuracy): {test_acc:.2f}%")
        if run is not None:
            run.summary["finetune/test_accuracy"] = test_acc
    finally:
        finish_wandb(run)


if __name__ == "__main__":
    main()
