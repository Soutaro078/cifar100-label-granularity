import argparse
import os

# Anaconda の MKL / OpenMP と PyTorch が衝突すると、学習中に
# Segmentation fault で落ちることがある。torch を読む前に固定する。
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader

from models.resnet import build_cifar_resnet18
from utils.cifar import load_split_dict
from utils.config import apply_overrides, load_yaml
from utils.metrics import (
    GranularityMeters,
    build_fine_to_coarse,
    move_fine_to_coarse,
    pretrain_wandb_metrics,
)
from utils.paths import (
    CONFIGS_DIR,
    DATA_ROOT,
    pretrain_checkpoint_path,
    pretrain_resume_path,
    pretrain_status_path,
)
from utils.tracking import finish_wandb, init_wandb, log_metrics, wandb_settings
from utils.transforms import cifar100_eval_transform, pretrain_train_transform


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-100 Pre-training")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIGS_DIR / "pretrain.yaml"),
        help="ハイパーパラメータの YAML",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["fine", "coarse"],
        help="学習モード: 'fine' (100クラス) または 'coarse' (20クラス)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="エポック数")
    parser.add_argument("--batch_size", type=int, default=None, help="バッチサイズ")
    parser.add_argument("--lr", type=float, default=None, help="学習率")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B のプロジェクト名")
    parser.add_argument("--no_wandb", action="store_true", help="W&B への記録を止める")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="途中保存から続ける（完了済みエポックの次から）",
    )
    return parser.parse_args()


def build_config(args) -> dict:
    cfg = apply_overrides(
        load_yaml(args.config),
        {
            "mode": args.mode,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
        },
    )
    if cfg["mode"] not in {"fine", "coarse"}:
        raise ValueError(f"mode は 'fine' または 'coarse' です: {cfg['mode']}")
    if int(cfg["epochs"]) < 1:
        raise ValueError(f"epochs は 1 以上である必要があります: {cfg['epochs']}")
    if cfg.get("scheduler") != "cosine":
        raise ValueError(
            "事前学習のスケジューラは cosine (CosineAnnealingLR) です: "
            f"{cfg.get('scheduler')}"
        )
    return cfg


def build_loader(dataset, batch_size, shuffle, num_workers):
    """CUDA を使ったあとで worker を fork すると、glibc の free() 破損で落ちる。

    学習ループは model.to("cuda") のあとで DataLoader を回す。Linux の既定は
    fork なので、worker が親の CUDA 状態を引き継いで Abort や Segmentation fault
    になる。spawn で起動し、エポックごとに作り直さない。
    """
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": False,
    }
    if num_workers > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"
        loader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **loader_kwargs)


def write_status(mode: str, message: str) -> None:
    path = pretrain_status_path(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(message + "\n", encoding="utf-8")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: _to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(value) for value in obj]
    return obj


def _atomic_torch_save(obj, path) -> None:
    """途中で落ちても 0 バイトの本番ファイルを残さない。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def save_progress(mode, epoch, model, optimizer, scheduler) -> None:
    """完了したエポックの重みと、再開用の最適化状態を残す。"""
    weights_path = pretrain_checkpoint_path(mode)
    resume_path = pretrain_resume_path(mode)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(_to_cpu(model.state_dict()), weights_path)
    _atomic_torch_save(
        {
            "epoch": epoch,
            "mode": mode,
            "model": _to_cpu(model.state_dict()),
            "optimizer": _to_cpu(optimizer.state_dict()),
            "scheduler": _to_cpu(scheduler.state_dict()),
        },
        resume_path,
    )


def load_progress(mode, model, optimizer, scheduler, device):
    resume_path = pretrain_resume_path(mode)
    if not resume_path.is_file() or resume_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"再開用ファイルがありません（空ファイルも含む）: {resume_path}"
        )
    checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
    if checkpoint.get("mode") != mode:
        raise ValueError(
            f"再開ファイルの mode が違います: {checkpoint.get('mode')} （指定は {mode}）"
        )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint["epoch"])


def evaluate(model, loader, criterion, device, track_derived_coarse, table, group_index):
    model.eval()
    meters = GranularityMeters(track_derived_coarse)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            meters.update(outputs, labels, criterion, table, group_index)
    return meters.summarize()


def format_epoch(epoch, epochs, train, val, lr, mode) -> str:
    if mode == "fine":
        return (
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {train['loss']:.4f} | Train Fine Acc: {train['accuracy']:.2f}% | "
            f"Train Coarse Acc: {train['coarse_accuracy']:.2f}% | "
            f"Val Loss: {val['loss']:.4f} | Val Fine Acc: {val['accuracy']:.2f}% | "
            f"Val Coarse Acc: {val['coarse_accuracy']:.2f}% | LR: {lr:.6f}"
        )
    return (
        f"Epoch [{epoch}/{epochs}] | "
        f"Train Loss: {train['loss']:.4f} | Train Coarse Acc: {train['accuracy']:.2f}% | "
        f"Val Loss: {val['loss']:.4f} | Val Coarse Acc: {val['accuracy']:.2f}% | LR: {lr:.6f}"
    )


def main():
    args = parse_args()
    cfg = build_config(args)
    mode = cfg["mode"]
    epochs = int(cfg["epochs"])
    batch_size = int(cfg["batch_size"])
    lr = float(cfg["lr"])
    track_derived_coarse = mode == "fine"
    num_classes = 100 if mode == "fine" else 20

    print(f"=== CIFAR-100 事前学習開始 (モード: {mode.upper()}, {num_classes}クラス) ===")
    print(
        f"epochs={epochs}, batch_size={batch_size}, lr={lr}, "
        f"momentum={cfg['momentum']}, weight_decay={cfg['weight_decay']}"
    )
    print(
        f"scheduler=CosineAnnealingLR(T_max={epochs}, eta_min={cfg['eta_min']}), "
        "augmentation=RandomCrop(32, padding=4)+RandomHorizontalFlip"
    )
    print("検証: CIFAR-100 の test split を val として毎エポック評価します。早期終了には使いません。")
    if track_derived_coarse:
        print("Fine 学習中は、予測クラスを公式の coarse 対応へ写して Coarse の Loss / Accuracy も記録します。")
    else:
        print("Coarse 学習の出力は 20 クラスのため、記録する評価は Coarse の Loss / Accuracy です。")

    trainset = torchvision.datasets.CIFAR100(
        root=str(DATA_ROOT),
        train=True,
        download=True,
        transform=pretrain_train_transform(),
    )
    valset = torchvision.datasets.CIFAR100(
        root=str(DATA_ROOT),
        train=False,
        download=True,
        transform=cifar100_eval_transform(),
    )

    train_dict = load_split_dict("train")
    table, groups = build_fine_to_coarse(train_dict)
    if mode == "coarse":
        trainset.targets = train_dict["coarse_labels"]
        valset.targets = load_split_dict("test")["coarse_labels"]

    num_workers = int(cfg["num_workers"])
    train_loader = build_loader(trainset, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = build_loader(valset, batch_size, shuffle=False, num_workers=num_workers)
    print(f"クラス数: {num_classes}, 訓練データ数: {len(trainset)}枚, 検証データ数: {len(valset)}枚")

    model = build_cifar_resnet18(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
    model = model.to(device)
    table, group_index = move_fine_to_coarse(table, groups, device)
    print(f"使用デバイス: {device}")
    print("stem: conv1 kernel=3 stride=1 padding=1, maxpool=Identity (32×32 用)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(cfg["eta_min"]),
    )

    wandb_on, wandb_project = wandb_settings(cfg, args.no_wandb, args.wandb_project)
    run = init_wandb(
        enabled=wandb_on,
        project=wandb_project,
        name=f"pretrain-{mode}",
        group="pretrain",
        job_type="pretrain",
        config={
            "stage": "pretrain",
            "mode": mode,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "momentum": float(cfg["momentum"]),
            "weight_decay": float(cfg["weight_decay"]),
            "num_workers": num_workers,
            "scheduler": cfg["scheduler"],
            "eta_min": float(cfg["eta_min"]),
            "architecture": "resnet18_cifar32",
            "num_classes": num_classes,
            "augmentation": "RandomCrop(32, padding=4)+RandomHorizontalFlip",
            "val_split": "cifar100_test",
            "device": str(device),
        },
    )

    try:
        start_epoch = 0
        if args.resume:
            start_epoch = load_progress(mode, model, optimizer, scheduler, device)
            print(f"チェックポイントから再開します。完了済みエポック: {start_epoch}", flush=True)
        if start_epoch >= epochs:
            print(f"すでに {start_epoch} エポックまで終わっています。", flush=True)
            return

        print("\n--- 学習ループ開始 ---", flush=True)
        print(f"進捗ファイル: {pretrain_status_path(mode)}", flush=True)
        for epoch_index in range(start_epoch, epochs):
            epoch = epoch_index + 1
            print(f"Epoch [{epoch}/{epochs}] を開始します。", flush=True)
            write_status(mode, f"epoch {epoch}/{epochs} train start")
            model.train()
            train_meters = GranularityMeters(track_derived_coarse)
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                train_meters.update(
                    outputs,
                    labels,
                    criterion,
                    table,
                    group_index,
                    task_loss=loss,
                )
                loss.backward()
                optimizer.step()

                if batch_idx % 50 == 0:
                    _synchronize(device)
                    write_status(mode, f"epoch {epoch}/{epochs} train batch {batch_idx}")

            _synchronize(device)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            write_status(mode, f"epoch {epoch}/{epochs} val start")
            train_stats = train_meters.summarize()
            val_stats = evaluate(
                model, val_loader, criterion, device, track_derived_coarse, table, group_index
            )
            _synchronize(device)
            print(format_epoch(epoch, epochs, train_stats, val_stats, current_lr, mode), flush=True)
            log_metrics(
                run,
                pretrain_wandb_metrics(mode, train_stats, val_stats, epoch, current_lr),
            )
            save_progress(mode, epoch, model, optimizer, scheduler)
            write_status(mode, f"epoch {epoch}/{epochs} saved")
            print(f"Epoch [{epoch}/{epochs}] の重みを保存しました。", flush=True)

        save_path = pretrain_checkpoint_path(mode)
        label = "Fine事前学習モデル" if mode == "fine" else "Coarse事前学習モデル"
        print(f"\n[完了] {label}を保存しました: {save_path}", flush=True)
        write_status(mode, f"done {epochs}/{epochs}")
        if run is not None:
            run.summary["checkpoint"] = str(save_path)
    finally:
        finish_wandb(run)


if __name__ == "__main__":
    main()
