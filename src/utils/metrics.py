import torch
import torch.nn as nn


def build_fine_to_coarse(split_dict: dict) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """画像ごとの fine / coarse ラベルから、fine クラス ID → coarse クラス ID の表を作る。

    CIFAR-100 では各 fine クラスはちょうど 1 つの coarse クラスに属し、
    各 coarse クラスは fine クラスを 5 個持つ。
    """
    mapping: dict[int, int] = {}
    for fine, coarse in zip(split_dict["fine_labels"], split_dict["coarse_labels"]):
        fine = int(fine)
        coarse = int(coarse)
        previous = mapping.get(fine)
        if previous is not None and previous != coarse:
            raise ValueError(f"fine label {fine} が複数の coarse に対応しています")
        mapping[fine] = coarse

    if set(mapping) != set(range(100)):
        raise ValueError(f"fine label が 0..99 を覆っていません: {len(mapping)} クラス")
    if set(mapping.values()) != set(range(20)):
        raise ValueError("coarse label が 0..19 を覆っていません")

    table = torch.tensor([mapping[i] for i in range(100)], dtype=torch.long)
    groups = []
    for coarse_id in range(20):
        indices = torch.where(table == coarse_id)[0]
        if indices.numel() != 5:
            raise ValueError(
                f"coarse {coarse_id} の fine クラス数が 5 ではありません: {indices.numel()}"
            )
        groups.append(indices)
    return table, groups


def move_fine_to_coarse(table: torch.Tensor, groups: list[torch.Tensor], device: torch.device):
    table = table.to(device)
    groups = [group.to(device) for group in groups]
    return table, groups


def coarse_logits_from_fine(logits: torch.Tensor, groups: list[torch.Tensor]) -> torch.Tensor:
    """100 クラスの logit を、同じ coarse に属する 5 クラスごとに logsumexp して 20 クラスにする。"""
    return torch.stack(
        [torch.logsumexp(logits.index_select(1, indices), dim=1) for indices in groups],
        dim=1,
    )


class GranularityMeters:
    """学習目標の Loss / Accuracy と、Fine モデルから導出した Coarse 指標を蓄積する。"""

    def __init__(self, track_derived_coarse: bool):
        self.track_derived_coarse = track_derived_coarse
        self.loss_sum = 0.0
        self.correct = 0
        self.coarse_loss_sum = 0.0
        self.coarse_correct = 0
        self.total = 0

    def update(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
        table: torch.Tensor | None = None,
        groups: list[torch.Tensor] | None = None,
    ) -> None:
        loss = criterion(outputs, labels)
        predicted = outputs.argmax(dim=1)
        batch_size = labels.size(0)
        self.loss_sum += loss.detach().item() * batch_size
        self.correct += predicted.eq(labels).sum().item()
        self.total += batch_size

        if not self.track_derived_coarse:
            return
        if table is None or groups is None:
            raise ValueError("Fine モデルの Coarse 指標には fine→coarse 対応表が必要です")

        coarse_labels = table[labels]
        coarse_predicted = table[predicted]
        coarse_logits = coarse_logits_from_fine(outputs, groups)
        coarse_loss = criterion(coarse_logits, coarse_labels)
        self.coarse_loss_sum += coarse_loss.detach().item() * batch_size
        self.coarse_correct += coarse_predicted.eq(coarse_labels).sum().item()

    def summarize(self) -> dict:
        if self.total == 0:
            raise ValueError("集計するバッチがありません")
        stats = {
            "loss": self.loss_sum / self.total,
            "accuracy": 100.0 * self.correct / self.total,
        }
        if self.track_derived_coarse:
            stats["coarse_loss"] = self.coarse_loss_sum / self.total
            stats["coarse_accuracy"] = 100.0 * self.coarse_correct / self.total
        return stats


def pretrain_wandb_metrics(mode: str, train: dict, val: dict, epoch: int, lr: float) -> dict:
    """エポックごとの記録。Accuracy はパーセント。

    Fine 学習では 100 クラス指標に加え、予測を coarse へ写した 20 クラス指標も入れる。
    Coarse 学習の出力は 20 クラスなので、Coarse 指標だけを入れる。
    """
    metrics = {
        "epoch": epoch,
        "lr": lr,
        "train/loss": train["loss"],
        "train/accuracy": train["accuracy"],
        "val/loss": val["loss"],
        "val/accuracy": val["accuracy"],
    }
    if mode == "fine":
        metrics.update(
            {
                "train/fine_loss": train["loss"],
                "train/fine_accuracy": train["accuracy"],
                "val/fine_loss": val["loss"],
                "val/fine_accuracy": val["accuracy"],
                "train/coarse_loss": train["coarse_loss"],
                "train/coarse_accuracy": train["coarse_accuracy"],
                "val/coarse_loss": val["coarse_loss"],
                "val/coarse_accuracy": val["coarse_accuracy"],
            }
        )
    elif mode == "coarse":
        metrics.update(
            {
                "train/coarse_loss": train["loss"],
                "train/coarse_accuracy": train["accuracy"],
                "val/coarse_loss": val["loss"],
                "val/coarse_accuracy": val["accuracy"],
            }
        )
    else:
        raise ValueError(f"mode は 'fine' または 'coarse' です: {mode}")
    return metrics
